"""사진 디코딩, 화질 측정, 설명 가능한 시각 특징 추출."""

from __future__ import annotations

import cv2
import numpy as np

from .models import ImageQuality, ImageSignature


class InvalidImageError(ValueError):
    """업로드 자료를 이미지로 읽을 수 없을 때 발생한다."""


def decode_image(data: bytes) -> np.ndarray:
    """JPG/PNG 바이트를 OpenCV BGR 이미지로 변환한다."""

    if not data:
        raise InvalidImageError("이미지 파일이 비어 있습니다. 다른 JPG/PNG 파일을 선택해 주세요.")
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise InvalidImageError("이미지를 읽을 수 없습니다. 손상되지 않은 JPG/PNG 파일인지 확인해 주세요.")
    return image


def measure_image_quality(image_bgr: np.ndarray) -> ImageQuality:
    """선명도·밝기·대비와 분석 신뢰도를 원본 사진에서 계산한다."""

    _validate_image(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    sharp_score = float(np.clip(sharpness / 140.0, 0.0, 1.0))
    exposure_score = float(np.clip(1.0 - abs(brightness - 128.0) / 118.0, 0.0, 1.0))
    contrast_score = float(np.clip(contrast / 52.0, 0.0, 1.0))
    reliability = 0.50 * sharp_score + 0.25 * exposure_score + 0.25 * contrast_score
    return ImageQuality(sharpness, brightness, contrast, float(np.clip(reliability, 0.0, 1.0)))


def extract_image_signature(image_bgr: np.ndarray) -> ImageSignature:
    """사진 중앙 영역에서 색·윤곽·저해상도 시각 패턴을 추출한다.

    첫 시제품은 동물 자동 분할 모델을 사용하지 않는다. 따라서 사용자는 동물이
    화면의 대부분을 차지하도록 잘라 올려야 하며, 배경색이 결과에 영향을 줄 수 있다.
    """

    _validate_image(image_bgr)
    normalized = _center_crop(image_bgr)
    normalized = cv2.resize(normalized, (192, 192), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)

    color_hist = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256]).flatten()
    color_hist = _normalize(color_hist)

    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    edge_hist, _ = np.histogram(angle, bins=12, range=(0.0, 360.0), weights=magnitude)
    edge_hist = _normalize(edge_hist.astype(np.float32))

    tiny = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    perceptual_bits = (tiny >= float(tiny.mean())).astype(np.uint8).flatten()
    dominant_colors = _dominant_color_names(hsv)

    return ImageSignature(color_hist, edge_hist, perceptual_bits, dominant_colors)


def aggregate_signatures(signatures: list[ImageSignature]) -> ImageSignature:
    """여러 기준 사진의 특징을 평균내 각도·조명 변화의 영향을 줄인다."""

    if not signatures:
        raise ValueError("기준 사진 특징이 한 개 이상 필요합니다.")
    color_hist = _normalize(np.mean([item.color_histogram for item in signatures], axis=0))
    edge_hist = _normalize(np.mean([item.edge_histogram for item in signatures], axis=0))
    bit_votes = np.mean([item.perceptual_bits for item in signatures], axis=0)
    perceptual_bits = (bit_votes >= 0.5).astype(np.uint8)

    color_counts: dict[str, int] = {}
    for signature in signatures:
        for color in signature.dominant_colors:
            color_counts[color] = color_counts.get(color, 0) + 1
    dominant = tuple(
        color for color, _ in sorted(color_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
    )
    return ImageSignature(color_hist, edge_hist, perceptual_bits, dominant)


def compare_signatures(reference: ImageSignature, candidate: ImageSignature) -> dict[str, float]:
    """두 이미지 특징의 단서별 유사도를 0~1로 반환한다."""

    color = _cosine_similarity(reference.color_histogram, candidate.color_histogram)
    texture = _cosine_similarity(reference.edge_histogram, candidate.edge_histogram)
    if reference.perceptual_bits.shape != candidate.perceptual_bits.shape:
        raise ValueError("비교할 시각 특징의 크기가 다릅니다.")
    visual = float(np.mean(reference.perceptual_bits == candidate.perceptual_bits))
    overall = 0.45 * color + 0.25 * texture + 0.30 * visual
    return {
        "overall": float(np.clip(overall, 0.0, 1.0)),
        "color": float(np.clip(color, 0.0, 1.0)),
        "texture": float(np.clip(texture, 0.0, 1.0)),
        "visual": float(np.clip(visual, 0.0, 1.0)),
    }


def _center_crop(image_bgr: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    margin_y = max(0, int(height * 0.08))
    margin_x = max(0, int(width * 0.08))
    crop = image_bgr[margin_y : height - margin_y, margin_x : width - margin_x]
    return crop if crop.size else image_bgr


def _dominant_color_names(hsv: np.ndarray) -> tuple[str, ...]:
    pixels = hsv.reshape(-1, 3)
    names = [_color_name(int(h), int(s), int(v)) for h, s, v in pixels[::4]]
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(name for name, _ in ordered[:3])


def _color_name(hue: int, saturation: int, value: int) -> str:
    if value < 45:
        return "검정"
    if saturation < 38 and value > 205:
        return "흰색"
    if saturation < 52:
        return "회색"
    if 7 <= hue < 24 and value < 165:
        return "갈색"
    if 8 <= hue < 32 and saturation < 105 and value >= 165:
        return "크림"
    if hue < 8 or hue >= 170:
        return "빨강"
    if hue < 23:
        return "주황"
    if hue < 36:
        return "노랑"
    if hue < 85:
        return "초록"
    if hue < 132:
        return "파랑"
    if hue < 165:
        return "보라"
    return "분홍"


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 1e-8 else np.zeros_like(values)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("비교할 특징의 크기가 다릅니다.")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _validate_image(image_bgr: np.ndarray) -> None:
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise InvalidImageError("3채널 컬러 이미지가 필요합니다.")
    if image_bgr.size == 0:
        raise InvalidImageError("이미지가 비어 있습니다.")

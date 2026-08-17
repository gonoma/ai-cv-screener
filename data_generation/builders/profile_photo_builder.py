import base64
import colorsys
import hashlib
import io
import os
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


class ProfilePhotoBuilder:
    _IMAGE_MODEL_NAME: str = "gemini-3.1-flash-image"
    _PROVIDER_CHAIN: tuple[str, ...] = ("gemini", "pollinations", "local")

    _PHOTO_PIXELS: int = 320
    _JPEG_QUALITY: int = 82

    _ATTIRE_OPTIONS: tuple[str, ...] = (
        "a dark tailored blazer",
        "an open-collar shirt",
        "a knitted jumper",
        "a smart casual jacket over a plain top",
        "a simple dark shirt",
    )
    _BACKDROP_OPTIONS: tuple[str, ...] = (
        "a plain light grey studio backdrop",
        "a warm white wall",
        "a soft beige backdrop",
        "a cool white studio backdrop",
    )

    def __init__(self) -> None:
        self._monogram_drawer = MonogramAvatarDrawer()

    def build_photo_prompt(self, presentation: str, appearance: str, index: int) -> str:
        """
        Who the person is comes from the record. Deriving appearance from the index
        can put a man's headshot on a woman's CV.
        """
        attire = self._ATTIRE_OPTIONS[index % len(self._ATTIRE_OPTIONS)]
        backdrop = self._BACKDROP_OPTIONS[index % len(self._BACKDROP_OPTIONS)]
        return (
            f"Candid headshot photograph of {presentation}, {appearance}, "
            f"wearing {attire}, photographed against {backdrop}. "
            "An ordinary working person photographed for a staff profile page: "
            "average everyday looks, plain and unremarkable, not a model, not "
            "glamorous, not styled, not conventionally attractive. "
            "Shot on a full-frame DSLR with an 85mm lens at f/2.8, soft directional "
            "window light from one side, gentle falloff, shallow depth of field with "
            "the background slightly out of focus. "
            "Natural untouched skin showing pores, blemishes, fine lines, stray "
            "hairs and slight facial asymmetry. Relaxed neutral expression, eyes to "
            "camera, head and shoulders, slightly off-centre framing. "
            "No retouching, no skin smoothing, no beauty filter, no slimming, no "
            "flattering angle, no glossy highlights. "
            "A real photograph, not an illustration, not a 3D render, not CGI. "
            "No text, no logo, no watermark, no border."
        )

    def build_photo(self, prompt: str, initials: str, seed: int) -> bytes:
        """Try each provider in turn, returning the first image anything produces.

        `seed` keeps two CVs apart when they share a name, and makes a retry
        produce a genuinely different image rather than the same one again.
        """
        start_at = os.environ.get("IMAGE_PROVIDER", "gemini").strip().lower()
        if start_at not in self._PROVIDER_CHAIN:
            raise ValueError(
                f"IMAGE_PROVIDER must be one of {list(self._PROVIDER_CHAIN)}, got {start_at!r}"
            )

        for provider in self._PROVIDER_CHAIN[self._PROVIDER_CHAIN.index(start_at) :]:
            try:
                if provider == "gemini":
                    return self._request_photo_from_gemini(prompt, seed)
                if provider == "pollinations":
                    return self._request_photo_from_pollinations(prompt, seed)
                return self._monogram_drawer.draw(initials, seed)
            except Exception as error:
                print(f"      {provider} failed ({type(error).__name__}: {error}), falling back")

        raise AssertionError("unreachable: the local rung does not fail")

    def normalise_to_small_jpeg(self, raw_image: bytes) -> bytes:
        image = Image.open(io.BytesIO(raw_image)).convert("RGB")
        image.thumbnail((self._PHOTO_PIXELS, self._PHOTO_PIXELS))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self._JPEG_QUALITY)
        return buffer.getvalue()

    @staticmethod
    def photo_as_data_uri(photo_path: Path) -> str:
        encoded = base64.b64encode(photo_path.read_bytes()).decode()
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def initials_from_name(name: str) -> str:
        name_parts = [part for part in name.replace("-", " ").split() if part]
        return "".join(part[0].upper() for part in name_parts[:2]) or "?"

    def _request_photo_from_gemini(self, prompt: str, seed: int) -> bytes:
        from google import genai

        interaction = genai.Client().interactions.create(
            model=self._IMAGE_MODEL_NAME, input=f"{prompt} Variation {seed}."
        )
        return base64.b64decode(interaction.output_image.data)

    @staticmethod
    def _request_photo_from_pollinations(prompt: str, seed: int) -> bytes:
        endpoint = (
            "https://image.pollinations.ai/prompt/"
            + urllib.parse.quote(prompt)
            + f"?width=512&height=512&nologo=true&seed={seed}"
        )
        # Explicit User-Agent is load-bearing: the endpoint 403s urllib's default.
        request = urllib.request.Request(endpoint, headers={"User-Agent": "cv-screening/0.1"})
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.read()


class MonogramAvatarDrawer:
    _IMAGE_PIXELS: int = 320
    _JPEG_QUALITY: int = 82
    _INK: tuple[int, int, int] = (28, 28, 28)

    # Above this, dark ink on the disc clears WCAG AAA at 7:1.
    _MINIMUM_DISC_LUMINANCE: float = 0.40

    def draw(self, initials: str, seed: int) -> bytes:
        image = Image.new("RGB", (self._IMAGE_PIXELS, self._IMAGE_PIXELS), self._disc_colour(seed))
        canvas = ImageDraw.Draw(image)
        monogram_font = ImageFont.load_default(size=self._IMAGE_PIXELS // 3)
        canvas.text(
            (self._IMAGE_PIXELS / 2, self._IMAGE_PIXELS / 2),
            initials,
            font=monogram_font,
            fill=self._INK,
            anchor="mm",
        )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self._JPEG_QUALITY)
        return buffer.getvalue()

    def _disc_colour(self, seed: int) -> tuple[int, int, int]:
        digest = hashlib.sha256(str(seed).encode()).hexdigest()
        rgb = self._hsv_to_rgb(
            int(digest[0:8], 16) / 0xFFFFFFFF,
            0.12 + int(digest[8:12], 16) / 0xFFFF * 0.20,
            0.80 + int(digest[12:16], 16) / 0xFFFF * 0.15,
        )
        return self._lighten_until_readable(rgb)

    def _lighten_until_readable(self, background: tuple[int, int, int]) -> tuple[int, int, int]:
        while self._relative_luminance(background) < self._MINIMUM_DISC_LUMINANCE:
            background = tuple(
                min(255, round(channel + (255 - channel) * 0.05)) for channel in background
            )
        return background

    @staticmethod
    def _relative_luminance(rgb: tuple[int, int, int]) -> float:
        linear = [
            channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in (component / 255 for component in rgb)
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    @staticmethod
    def _hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
        return tuple(
            int(component * 255) for component in colorsys.hsv_to_rgb(hue, saturation, value)
        )

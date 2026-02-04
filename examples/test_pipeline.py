"""Test image generation using pipeline mode."""

from image_gen import ImageGenerator

print("Initializing ImageGenerator (pipeline mode)...")
gen = ImageGenerator(mode="pipeline")

print("Generating image...")
image = gen.generate(
    prompt="A serene mountain landscape at sunset with a crystal clear lake",
    seed=42,
)

image.save("output/test_pipeline.png")
print("Image saved to: output/test_pipeline.png")

from PIL import Image
import numpy as np

img = Image.open("assets/logo.png").convert("RGBA")
data = np.array(img)

r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
# Distance from pure white
dist = np.sqrt((255 - r.astype(float))**2 + (255 - g.astype(float))**2 + (255 - b.astype(float))**2)

# Make anything very close to white (dist < 80) partially or fully transparent
mask = np.clip(dist / 80.0 * 255, 0, 255).astype(np.uint8)
data[:,:,3] = np.minimum(a, mask)

Image.fromarray(data).save("assets/logo.png")
print("Background removed.")

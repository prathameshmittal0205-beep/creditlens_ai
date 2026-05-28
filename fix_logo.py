from PIL import Image
import numpy as np

img = Image.open("C:\\Users\\pratm\\.gemini\\antigravity\\brain\\582a6b87-95e2-4917-8ba0-9e5cae6cee4d\\media__1779995748674.png").convert("RGBA")
data = np.array(img)

r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
# Distance from white
dist = np.sqrt((255 - r.astype(float))**2 + (255 - g.astype(float))**2 + (255 - b.astype(float))**2)

# Make anything remotely white transparent to ensure no patchiness
# The logo itself is blue, so it has a large distance from white (e.g., > 100).
mask = np.ones_like(a) * 255

# Apply a smooth alpha based on distance for anti-aliasing
# If dist < 30 (very white), alpha = 0
# If dist > 150 (very colored), alpha = 255
# In between, linear
mask = np.clip((dist - 30) / 120.0 * 255, 0, 255).astype(np.uint8)

# Make sure we don't increase alpha if original had lower alpha
data[:,:,3] = np.minimum(a, mask)

Image.fromarray(data).save("assets/logo.png")
print("Done")

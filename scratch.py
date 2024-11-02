import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Read images
images = []
for i in range(1,11):  # Assuming there are 10 images named canvas_0.png to canvas_9.png
    img = cv2.imread(f'/home/luisamao/gaussian_terrain_mapping/canvas_{i}.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img_rgb)

# Create figure
fig = plt.figure()

# Initialize animation
ims = []
for img in images:
    im = plt.imshow(img, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)

# Save animation
ani.save('/home/luisamao/gaussian_terrain_mapping/animation.mp4')

plt.show()
import pygame

# 1. Initialize Pygame
pygame.init()

# 2. Set up the display window
# Define window dimensions
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Set the window title
pygame.display.set_caption("Green Pixel Example")

# Define colors (RGB tuples)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# 3. Game loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        # If the user clicks the close button
        if event.type == pygame.QUIT:
            running = False

    # 4. Drawing operations
    # Fill the background with black (or any other color)
    screen.fill(BLACK)

    # Draw a single green pixel at coordinates (100, 150)
    # The draw.pixel() method expects the surface, color, and position
    # The position is an (x, y) tuple, where (0,0) is the top-left corner.
    pygame.draw.circle(screen, GREEN, (100, 150), 1) # A circle with radius 1 is a pixel

    # Alternative way to draw a pixel using set_at (less common for drawing directly, more for manipulating existing pixels)
    # screen.set_at((101, 150), GREEN)


    # 5. Update the display
    # This makes everything you've drawn visible on the screen
    pygame.display.flip() # or pygame.display.update()

# 6. Quit Pygame
pygame.quit()
print("Pygame window closed.")
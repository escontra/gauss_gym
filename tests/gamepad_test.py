import pygame

# Initialize Pygame and its joystick module
pygame.init()
pygame.joystick.init()

# Check for joystick(s)
joystick_count = pygame.joystick.get_count()
print(f"Number of joysticks: {joystick_count}")
if joystick_count == 0:
    print("No joystick detected.")
else:
    # Initialize the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Joystick initialized:", joystick.get_name())

# Main loop to poll events
running = True
while running:
    # Process event queue
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Handle axis motion events
        if event.type == pygame.JOYAXISMOTION:
            axis = event.axis
            value = event.value
            print(f"Axis {axis} moved to {value}")

        # Handle button press events
        elif event.type == pygame.JOYBUTTONDOWN:
            print(f"Button {event.button} pressed")
        elif event.type == pygame.JOYBUTTONUP:
            print(f"Button {event.button} released")

        # Handle hat (d-pad) motion events
        elif event.type == pygame.JOYHATMOTION:
            print(f"Hat moved to {event.value}")

    # A short delay to avoid busy-waiting
    pygame.time.wait(10)

pygame.quit()

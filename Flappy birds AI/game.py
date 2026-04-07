import pygame
import sys
import random
from dataclasses import dataclass
from typing import Sequence


# --- Game Configuration and State ---

@dataclass
class GameConfig:
    win_width: int = 551
    win_height: int = 720
    fps: int = 60
    scroll_speed: int = 1

    bg_color: tuple[int, int, int] = (0, 0, 0)
    text_color: tuple[int, int, int] = (255, 255, 255)

    bird_start_position: tuple[int, int] = (100, 250)
    ground_y: int = 520
    score_position: tuple[int, int] = (20, 20)

    gravity: float = 0.5
    terminal_velocity: float = 7.0
    flap_strength: float = -7.0
    rotation_factor: float = -7.0

    pipe_spawn_delay_min: int = 180
    pipe_spawn_delay_max: int = 250
    pipe_gap_min: int = 90
    pipe_gap_max: int = 130
    pipe_y_top_min: int = -600
    pipe_y_top_max: int = -480


@dataclass
class GameState:
    score: int = 0
    game_stopped: bool = True


config: GameConfig = GameConfig()
state: GameState = GameState()

# --- Pygame Initialization ---

pygame.init()
clock: pygame.time.Clock = pygame.time.Clock()

window: pygame.Surface = pygame.display.set_mode((config.win_width, config.win_height))
pygame.display.set_caption("Flappy Bird Clone")

bird_images: list[pygame.Surface] = [
    pygame.image.load("assets/bird_down.png"),
    pygame.image.load("assets/bird_mid.png"),
    pygame.image.load("assets/bird_up.png")
]
skyline_image: pygame.Surface = pygame.image.load("assets/background.png")
ground_image: pygame.Surface = pygame.image.load("assets/ground.png")
top_pipe_image: pygame.Surface = pygame.image.load("assets/pipe_top.png")
bottom_pipe_image: pygame.Surface = pygame.image.load("assets/pipe_bottom.png")
game_over_image: pygame.Surface = pygame.image.load("assets/game_over.png")
start_image: pygame.Surface = pygame.image.load("assets/start.png")

font: pygame.font.Font = pygame.font.SysFont('Segoe', 26)


# --- Classes ---

class Bird(pygame.sprite.Sprite):
    def __init__(self) -> None:
        super().__init__()
        self.image: pygame.Surface = bird_images[0]
        self.rect: pygame.Rect = self.image.get_rect()
        self.rect.center = config.bird_start_position
        self.image_index: int = 0
        self.vel: float = 0.0
        self.flap: bool = False
        self.alive: bool = True

    def update(self, user_input: Sequence[bool]) -> None:
        if self.alive:
            self.image_index += 1
        if self.image_index >= 30:
            self.image_index = 0
        self.image = bird_images[self.image_index // 10]

        self.vel += config.gravity
        if self.vel > config.terminal_velocity:
            self.vel = config.terminal_velocity

        if self.rect.y < config.ground_y - 20:
            self.rect.y += int(self.vel)

        if self.vel == 0:
            self.flap = False

        self.image = pygame.transform.rotate(self.image, self.vel * config.rotation_factor)

        if user_input[pygame.K_SPACE] and not self.flap and self.rect.y > 0 and self.alive:
            self.flap = True
            self.vel = config.flap_strength


class Pipe(pygame.sprite.Sprite):
    def __init__(self, x: int, y: int, image: pygame.Surface, pipe_type: str) -> None:
        super().__init__()
        self.image: pygame.Surface = image
        self.rect: pygame.Rect = self.image.get_rect()
        self.rect.x, self.rect.y = x, y
        self.enter: bool = False
        self.exit: bool = False
        self.passed: bool = False
        self.pipe_type: str = pipe_type

    def update(self) -> None:
        self.rect.x -= config.scroll_speed
        if self.rect.x <= -config.win_width:
            self.kill()

        if self.pipe_type == 'bottom':
            if config.bird_start_position[0] > self.rect.topleft[0] and not self.passed:
                self.enter = True
            if config.bird_start_position[0] > self.rect.topright[0] and not self.passed:
                self.exit = True
            if self.enter and self.exit and not self.passed:
                self.passed = True
                state.score += 1


class Ground(pygame.sprite.Sprite):
    def __init__(self, x: int, y: int) -> None:
        super().__init__()
        self.image: pygame.Surface = ground_image
        self.rect: pygame.Rect = self.image.get_rect()
        self.rect.x, self.rect.y = x, y

    def update(self) -> None:
        self.rect.x -= config.scroll_speed
        if self.rect.x <= -config.win_width:
            self.kill()


# --- Helper Functions ---

def spawn_entities(ground: pygame.sprite.Group, pipes: pygame.sprite.Group, pipe_timer: int, bird_alive: bool) -> int:
    """Manages the spawning of new pipes and ground. Returns the updated pipe timer."""
    # Ground
    if len(ground) <= 2:
        ground.add(Ground(config.win_width, config.ground_y))

    # Pipes
    if pipe_timer <= 0 and bird_alive:
        x_pos: int = 550
        y_top: int = random.randint(config.pipe_y_top_min, config.pipe_y_top_max)
        y_bottom: int = y_top + random.randint(config.pipe_gap_min, config.pipe_gap_max) + bottom_pipe_image.get_height()

        pipes.add(Pipe(x_pos, y_top, top_pipe_image, 'top'))
        pipes.add(Pipe(x_pos, y_bottom, bottom_pipe_image, 'bottom'))

        pipe_timer = random.randint(config.pipe_spawn_delay_min, config.pipe_spawn_delay_max)

    return pipe_timer - 1


def handle_collisions(bird: pygame.sprite.GroupSingle, pipes: pygame.sprite.Group, ground: pygame.sprite.Group) -> bool:
    """Checks for collisions and kills the bird. Returns True if the ground is hit."""
    bird_sprite: pygame.sprite.Sprite = bird.sprites()[0]
    collision_pipes: list[pygame.sprite.Sprite] = pygame.sprite.spritecollide(bird_sprite, pipes, False)
    collision_ground: list[pygame.sprite.Sprite] = pygame.sprite.spritecollide(bird_sprite, ground, False)

    if collision_pipes or collision_ground:
        bird_sprite.alive = False

    return bool(collision_ground)


def draw_game(window: pygame.Surface, bird: pygame.sprite.GroupSingle, pipes: pygame.sprite.Group, ground: pygame.sprite.Group, hit_ground: bool) -> None:
    """Handles drawing all game elements on the screen."""
    window.fill(config.bg_color)
    window.blit(skyline_image, (0, 0))

    pipes.draw(window)
    ground.draw(window)
    bird.draw(window)

    score_text: pygame.Surface = font.render(f'Score: {state.score}', True, config.text_color)
    window.blit(score_text, config.score_position)

    if hit_ground:
        window.blit(game_over_image, (config.win_width // 2 - game_over_image.get_width() // 2,
                                      config.win_height // 2 - game_over_image.get_height() // 2))


# --- Main Loops ---

def main() -> None:
    # Initialize game objects
    bird: pygame.sprite.GroupSingle = pygame.sprite.GroupSingle()
    bird.add(Bird())
    pipes: pygame.sprite.Group = pygame.sprite.Group()
    ground: pygame.sprite.Group = pygame.sprite.Group()
    ground.add(Ground(0, config.ground_y))

    pipe_timer: int = 0
    run: bool = True

    while run:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        user_input: Sequence[bool] = pygame.key.get_pressed()


        pipe_timer = spawn_entities(ground, pipes, pipe_timer, bird.sprite.alive)

        # Update physics and movement
        if bird.sprite.alive:
            pipes.update()
            ground.update()
        bird.update(user_input)

        # Collision Detection
        hit_ground: bool = handle_collisions(bird, pipes, ground)

        # Render frame
        draw_game(window, bird, pipes, ground, hit_ground)

        #  Restart game (if the bird hit the ground and 'R' is pressed)
        if hit_ground and user_input[pygame.K_r]:
            state.score = 0
            break

        clock.tick(config.fps)
        pygame.display.update()


def menu() -> None:
    while state.game_stopped:
        # System Event Handling (Menu)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        window.fill(config.bg_color)
        window.blit(skyline_image, (0, 0))
        window.blit(ground_image, (0, config.ground_y))
        window.blit(bird_images[0], config.bird_start_position)
        window.blit(start_image, (config.win_width // 2 - start_image.get_width() // 2,
                                  config.win_height // 2 - start_image.get_height() // 2))

        user_input: Sequence[bool] = pygame.key.get_pressed()
        if user_input[pygame.K_SPACE]:
            state.game_stopped = False
            main()
            state.game_stopped = True

        pygame.display.update()


# --- Entry Point ---
if __name__ == "__main__":
    menu()

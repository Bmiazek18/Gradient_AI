import os
import sys
import pickle
import random
import pygame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from dataclasses import dataclass
from typing import Tuple


# --- Game Configuration ---
@dataclass
class GameConfig:
    """
    Configuration class containing all physical, visual, and environmental constants
    used throughout the game and training process.
    """
    win_width: int = 551
    win_height: int = 720
    fps: int = 60
    scroll_speed: int = 5

    bird_start_position: tuple[int, int] = (100, 250)
    ground_y: int = 520

    gravity: float = 0.5
    terminal_velocity: float = 7.0
    flap_strength: float = -7.0

    pipe_spawn_delay_min: int = 80
    pipe_spawn_delay_max: int = 120
    pipe_gap_min: int = 120
    pipe_gap_max: int = 150
    pipe_y_top_min: int = -600
    pipe_y_top_max: int = -480


config = GameConfig()
MODEL_FILENAME = "flappy_model_2501_ep.pkl"

# --- Pygame Initialization ---
pygame.init()
font = pygame.font.SysFont('Segoe', 36)

try:
    bird_images = [
        pygame.image.load("assets/bird_down.png"),
        pygame.image.load("assets/bird_mid.png"),
        pygame.image.load("assets/bird_up.png")
    ]
    top_pipe_image = pygame.image.load("assets/pipe_top.png")
    bottom_pipe_image = pygame.image.load("assets/pipe_bottom.png")
    ground_image = pygame.image.load("assets/ground.png")
    background_image = pygame.image.load("assets/background.png")
except FileNotFoundError:
    print("[INFO] No image files found in 'assets/'. Using placeholder rectangles.")
    bird_images = [pygame.Surface((34, 24)) for _ in range(3)]
    for b in bird_images: b.fill((255, 255, 0))
    top_pipe_image = pygame.Surface((52, 600));
    top_pipe_image.fill((0, 255, 0))
    bottom_pipe_image = pygame.Surface((52, 600));
    bottom_pipe_image.fill((0, 255, 0))
    ground_image = pygame.Surface((config.win_width, 200));
    ground_image.fill((139, 69, 19))
    background_image = pygame.Surface((config.win_width, config.win_height));
    background_image.fill((135, 206, 235))


# --- Agent and Game Classes ---
class QAgent:
    """
    Q-Learning agent responsible for learning the game strategy through exploration
    and exploitation, updating its Q-table based on received rewards.
    """

    def __init__(self, mode="play"):
        """
        Initializes the agent's parameters based on the selected mode.

        Args:
            mode (str): 'train' for learning with exploration, 'play' for strictly
                        using the learned policy (exploitation only).
        """
        self.q_table = {}
        self.gamma = 0.99

        if mode == "train":
            self.alpha = 0.1
            self.alpha_min = 0.01
            self.alpha_decay = 0.9999
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.999
        else:
            self.alpha = 0.0
            self.epsilon = 0.0

    def get_q(self, state, action) -> float:

        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state) -> int:
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state (Tuple): The current state of the environment.

        Returns:
            int: 1 for flap, 0 for do nothing.
        """
        # Exploration vs Exploitation
        if random.random() < self.epsilon:
            return 1 if random.random() < 0.08 else 0  # 8% chance to flap during exploration
        if self.get_q(state, 1) > self.get_q(state, 0):
            return 1
        return 0

    def update(self, state, action, reward, next_state, done):
        """
        Updates the Q-table using the Bellman equation based on the agent's experience.

        Args:
            state (Tuple): The state before the action.
            action (int): The action taken.
            reward (float): The reward received after taking the action.
            next_state (Tuple): The state resulting from the action.
            done (bool): True if the action resulted in game over, False otherwise.
        """
        max_future_q = 0.0 if done else max(self.get_q(next_state, 0), self.get_q(next_state, 1))
        current_q = self.get_q(state, action)
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        """
        Gradually reduces the exploration rate (epsilon) and learning rate (alpha)
        to stabilize learning over time.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.alpha > self.alpha_min:
            self.alpha *= self.alpha_decay

    def save_model(self, filename):

        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"[SAVE] Model successfully saved to: {filename}")

    def load_model(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"[LOAD] Loaded {len(self.q_table)} states from {filename}.")
            return True
        except FileNotFoundError:
            print(f"[ERROR] Could not find {filename}!")
            return False


class Bird(pygame.sprite.Sprite):
    """
    Represents the player-controlled Bird, handling its physics, rotation,
    and hitbox logic independently of visual presentation.
    """

    def __init__(self):
        super().__init__()
        self.frames = bird_images
        self.frame_index = 1
        self.image = self.frames[self.frame_index]
        self.rect = self.image.get_rect()
        self.hitbox = pygame.Rect(0, 0, 30, 20)
        self.hitbox.center = config.bird_start_position
        self.vel = 0.0

    def update(self, action: int):
        """
        Updates the bird's vertical velocity, position, and visual rotation.

        Args:
            action (int): 1 if the bird flaps, 0 otherwise.
        """
        # Physics
        self.vel += config.gravity
        if self.vel > config.terminal_velocity:
            self.vel = config.terminal_velocity
        if self.hitbox.y < config.ground_y - 20:
            self.hitbox.y += int(self.vel)

        # Flap logic
        if action == 1 and self.hitbox.y > 0:
            self.vel = config.flap_strength
            self.frame_index = 2
        else:
            self.frame_index = 1

        # Visual rotation based on velocity
        base_image = self.frames[int(self.frame_index)]
        self.image = pygame.transform.rotate(base_image, self.vel * -3)
        self.rect = self.image.get_rect(center=self.hitbox.center)


class Pipe(pygame.sprite.Sprite):
    """
    Represents a stationary pipe obstacle that moves leftwards across the screen.
    """

    def __init__(self, x, y, image, pipe_type):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = x, y
        self.passed = False
        self.pipe_type = pipe_type

    def update(self):
        """Moves the pipe left based on scroll speed and removes it if off-screen."""
        self.rect.x -= config.scroll_speed
        if self.rect.x <= -config.win_width:
            self.kill()


class Ground(pygame.sprite.Sprite):
    """
    Represents the scrolling ground surface at the bottom of the screen.
    """

    def __init__(self, x, y):
        super().__init__()
        self.image = ground_image
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = x, y

    def update(self):
        """Moves the ground left to create a scrolling effect."""
        self.rect.x -= config.scroll_speed
        if self.rect.x <= -config.win_width:
            self.kill()


# --- State Function ---
def get_state(bird: Bird, pipes: pygame.sprite.Group) -> Tuple[int, int, int]:
    """
    Observes the game environment and translates it into a discrete, finite state.
    This limits the state space and allows the Q-Table to converge efficiently.

    Args:
        bird (Bird): The player entity.
        pipes (pygame.sprite.Group): All active pipe obstacles.

    Returns:
        Tuple[int, int, int]: The discretized state (dx, dy, vel).
    """
    bird_x = bird.hitbox.x
    bottom_pipe = None
    top_pipe = None

    # Find the closest upcoming pipe
    for p in pipes:
        if p.pipe_type == 'bottom' and p.rect.right > bird_x:
            if bottom_pipe is None or p.rect.x < bottom_pipe.rect.x:
                bottom_pipe = p

    # If no pipe is visible
    if bottom_pipe is None:
        return (14, 0, int(bird.vel // 2))

    # Match the corresponding top pipe
    for p in pipes:
        if p.pipe_type == 'top' and p.rect.x == bottom_pipe.rect.x:
            top_pipe = p
            break

    # Calculate the center of the gap
    if top_pipe and bottom_pipe:
        gap_center_y = (top_pipe.rect.bottom + bottom_pipe.rect.top) // 2
    else:
        gap_center_y = bottom_pipe.rect.top - (config.pipe_gap_min // 2)

    # Discretize state variables (Binning) to reduce state space
    dx = (bottom_pipe.rect.x - bird_x) // 40
    dy = (gap_center_y - bird.hitbox.centery) // 40
    vel = int(bird.vel // 2)

    # Clamp values to prevent unbounded state growth
    dx = max(0, min(dx, 14))
    dy = max(-15, min(dy, 15))
    return (dx, dy, vel)


def spawn_entities(ground, pipes, pipe_timer, bird_alive):
    """
    Manages the continuous generation of ground tiles and pipe pairs.

    Args:
        ground (pygame.sprite.Group): Group managing ground tiles.
        pipes (pygame.sprite.Group): Group managing pipe obstacles.
        pipe_timer (int): Countdown until the next pipe spawn.
        bird_alive (bool): Determines if new pipes should spawn.

    Returns:
        int: The updated pipe_timer value.
    """
    if len(ground) <= 2:
        ground.add(Ground(config.win_width, config.ground_y))
    if pipe_timer <= 0 and bird_alive:
        x_pos = 550
        y_top = random.randint(config.pipe_y_top_min, config.pipe_y_top_max)
        y_bottom = y_top + random.randint(config.pipe_gap_min, config.pipe_gap_max) + bottom_pipe_image.get_height()
        pipes.add(Pipe(x_pos, y_top, top_pipe_image, 'top'))
        pipes.add(Pipe(x_pos, y_bottom, bottom_pipe_image, 'bottom'))
        pipe_timer = random.randint(config.pipe_spawn_delay_min, config.pipe_spawn_delay_max)
    return pipe_timer - 1


# --- Core Modules ---

def train():
    """
    Executes the training loop in headless mode (no visual rendering) for maximum speed.
    The agent learns through trial and error over thousands of episodes.
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"  # Hide Pygame window for faster training
    agent = QAgent(mode="train")

    episodes = 20000
    print(f"Starting background training ({episodes} episodes). Please wait...")

    for episode in range(episodes):
        bird = pygame.sprite.GroupSingle(Bird())
        pipes = pygame.sprite.Group()
        ground = pygame.sprite.Group(Ground(0, config.ground_y))
        pipe_timer = 0
        score = 0

        while bird.sprite.alive:
            pygame.event.pump()
            current_state = get_state(bird.sprite, pipes)
            action = agent.choose_action(current_state)

            pipe_timer = spawn_entities(ground, pipes, pipe_timer, bird.sprite.alive)
            pipes.update()
            ground.update()
            bird.sprite.update(action)

            # Collision Detection
            collision = False
            for p in pipes:
                if bird.sprite.hitbox.colliderect(p.rect): collision = True
            for g in ground:
                if bird.sprite.hitbox.colliderect(g.rect): collision = True
            if bird.sprite.hitbox.y < 0: collision = True  # Hit ceiling

            if collision: bird.sprite.alive = False

            # Score tracking
            passed = False
            for p in pipes:
                if p.pipe_type == 'bottom' and bird.sprite.hitbox.left > p.rect.right and not p.passed:
                    p.passed = True
                    score += 1
                    passed = True

            # Reward Shaping
            if not bird.sprite.alive:
                reward = -1000
            elif passed:
                reward = 100
            else:
                reward = 0.1

            if action == 1: reward -= 0.2  # Small penalty for flapping to encourage falling

            next_state = get_state(bird.sprite, pipes)
            agent.update(current_state, action, reward, next_state, not bird.sprite.alive)

        agent.decay_epsilon()

        # Logging progress
        if (episode + 1) % 1000 == 0:
            print(
                f"Ep: {episode + 1} | Last Score: {score} | Eps: {agent.epsilon:.3f} | Alpha: {agent.alpha:.3f} | States: {len(agent.q_table)}")

    agent.save_model(MODEL_FILENAME)
    print("Training complete!")


def play():
    """
    Initializes a game window and lets the pre-trained agent play automatically.
    Exploration is disabled; the agent strictly follows its learned policy.
    """
    agent = QAgent(mode="play")
    if not agent.load_model(MODEL_FILENAME): return

    window = pygame.display.set_mode((config.win_width, config.win_height))
    pygame.display.set_caption("AI Flappy Bird - Agent Playing")
    clock = pygame.time.Clock()

    bird = pygame.sprite.GroupSingle(Bird())
    pipes = pygame.sprite.Group()
    ground = pygame.sprite.Group(Ground(0, config.ground_y))
    pipe_timer = 0
    score = 0

    print("Game started. Agent is relying entirely on the trained model.")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit();
                return

        current_state = get_state(bird.sprite, pipes)
        action = agent.choose_action(current_state)

        pipe_timer = spawn_entities(ground, pipes, pipe_timer, True)
        pipes.update()
        ground.update()
        bird.sprite.update(action)

        collision = False
        for p in pipes:
            if bird.sprite.hitbox.colliderect(p.rect): collision = True
        for g in ground:
            if bird.sprite.hitbox.colliderect(g.rect): collision = True
        if bird.sprite.hitbox.y < 0: collision = True

        for p in pipes:
            if p.pipe_type == 'bottom' and bird.sprite.hitbox.left > p.rect.right and not p.passed:
                p.passed = True
                score += 1

        if collision:
            print(f"Agent failed! Game over with score: {score}")
            score = 0
            pipes.empty()
            bird.sprite.hitbox.center = config.bird_start_position
            bird.sprite.vel = 0

        # Rendering
        window.blit(background_image, (0, 0))
        pipes.draw(window)
        ground.draw(window)
        window.blit(bird.sprite.image, bird.sprite.rect)
        window.blit(font.render(f"AI Score: {score}", True, (255, 255, 255)), (20, 20))
        pygame.display.update()
        clock.tick(config.fps)


def plot_agent_strategy():
    """
    Generates a heatmap visualization of the agent's learned policy using Matplotlib.
    Distinguishes between flap, fall, and unknown (unvisited) states.
    """
    agent = QAgent(mode="play")
    if not agent.load_model(MODEL_FILENAME): return

    dy_range = range(-15, 16)
    vel_range = range(-7, 8)
    fixed_dx = 4

    matrix = np.zeros((len(dy_range), len(vel_range)))

    for i, dy in enumerate(dy_range):
        for j, vel in enumerate(vel_range):
            state = (fixed_dx, dy, vel)

            has_action_0 = (state, 0) in agent.q_table
            has_action_1 = (state, 1) in agent.q_table

            if not has_action_0 and not has_action_1:
                matrix[i, j] = -1  # Unknown State
            else:
                if agent.get_q(state, 1) > agent.get_q(state, 0):
                    matrix[i, j] = 1  # Flap
                else:
                    matrix[i, j] = 0  # Fall

    plt.figure(figsize=(10, 8))
    cmap = ListedColormap(['#bdc3c7', '#e74c3c', '#2ecc71'])
    sns.heatmap(matrix, xticklabels=vel_range, yticklabels=dy_range, cmap=cmap, cbar=False)

    plt.title(f"Agent Strategy (Pipe distance dx={fixed_dx})\nGray=UNKNOWN | Green=FLAP | Red=FALL")
    plt.xlabel("Bird Velocity (vel // 2)")
    plt.ylabel("Height from gap center (dy // 40)")

    legend_elements = [Patch(facecolor='#2ecc71', label='Flap (Action 1)'),
                       Patch(facecolor='#e74c3c', label='Fall (Action 0)'),
                       Patch(facecolor='#bdc3c7', label='Unknown State')]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.gca().invert_yaxis()
    plt.show()


# --- ENTRY POINT (Main Menu) ---
if __name__ == "__main__":
    print("\n--- AI FLAPPY BIRD MENU ---")
    print("1. Train new Agent (Fast background mode)")
    print("2. Watch trained Agent play")
    print("3. Generate strategy map (Plot)")
    print("0. Exit")

    choice = input("\nSelect an option (0-3): ")

    if choice == '1':
        train()
    elif choice == '2':
        play()
    elif choice == '3':
        plot_agent_strategy()
    else:
        sys.exit()

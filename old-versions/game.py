import random


class Bird:
    def __init__(self):
        self.x = 0.0
        self.y = 100.0
        self.yvel = 0.0
        self.dead = False
        self.width = 17
        self.height = 12


class Pipe:
    def __init__(self, x):
        self.x = x
        self.y = random.randint(79, 135)
        self.counted = False
        self.width = 26
        self.height = 47  # height of the space top and bottom pipe


class Game:
    def __init__(self):
        self.restart()

    def restart(self):
        self.pipes = [Pipe(200), Pipe(300)]
        self.bird = Bird()
        self.score = 0

    def space(self):
        if self.bird.dead:
            self.restart()
        self.jump()

    def jump(self):
        if not self.bird.dead:
            self.bird.yvel = 2.5

    def tick(self, delta_time):
        if self.bird.y < 16:
            self.bird.dead = True

        self.bird.yvel = max(-4, self.bird.yvel - delta_time * 8)
        self.bird.y = max(13, self.bird.y + self.bird.yvel)

        if not self.bird.dead:
            self.bird.x += 1

            if self.pipes[0].x - self.bird.x + 80 < 0:
                pipe = Pipe(self.pipes[0].x + 200)
                self.pipes = [self.pipes[1], pipe]

            for pipe in self.pipes:
                if (
                    pipe.x < self.bird.x < pipe.x + pipe.width
                    or pipe.x < self.bird.x + self.bird.width < pipe.x + pipe.width
                ):
                    if (
                        self.bird.y + self.bird.height / 2 > pipe.y
                        or self.bird.y < pipe.y - pipe.height
                    ):
                        self.bird.dead = True
                        break
                    if not pipe.counted:
                        self.score += 1
                        pipe.counted = True

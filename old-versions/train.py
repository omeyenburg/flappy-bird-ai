import game
import ai


def train(agent):
    g = game.Game()
    score = 0

    while g.bird.x < 3000 and not g.bird.dead:
        pipes = {abs(pipe.x - g.bird.x): pipe for pipe in g.pipes}
        next_pipe = pipes[min(pipes.keys())]

        if not (
            g.bird.y + g.bird.height / 2 > next_pipe.y
            or g.bird.y < next_pipe.y - next_pipe.height
        ):
            score += 2

        inputs = [g.bird.y, g.bird.yvel, next_pipe.x - g.bird.x, next_pipe.y]
        output = agent.run(inputs)
        score += abs(output[0] - 0.5)

        if output[0] > 0.5:
            g.jump()

        g.tick(1 / 60)

    score += g.score * 10
    return score


def main():
    rlm = ai.ReinforcementLearningModel(
        train, 30, ["y", "yvel", "xpipe", "ypipe"], ["jump"], [4, 4, 4], "sigmoid"
    )
    rlm.train()


if __name__ == "__main__":
    main()

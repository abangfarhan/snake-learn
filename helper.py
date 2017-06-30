from PIL import Image
import numpy as np

# def get_screen(browser):
#     """ 
#     Return a cropped 2D binary array
#     representation of the game.
#     Include border and the score
#     """
#     browser.save_screenshot('snake.png')
#     g = browser.find_element_by_id('game')

#     xtop = g.location['x'] + 12  # crop game padding
#     ytop = g.location['y'] + 12
#     # should scores be cut too?

#     xbot = g.location['x'] + g.size['width'] - 12
#     ybot = g.location['y'] + g.size['height'] - 12

#     im = Image.open('snake.png')
#     im = im.crop((xtop, ytop, xbot, ybot))
#     # im.save('snake.png')
#     # -----------------------
#     X = np.array([item[0] for item in im.getdata()]).reshape(im.height, im.width)
#     X[X <= 100] = 1
#     X[X > 100] = 0
#     return X

def get_screen(browser):
    """ 
    Return a cropped 2D binary array
    representation of the game.
    Exclude the border and the score
    """
    browser.save_screenshot('snake.png')
    g = browser.find_element_by_id('game')

    xtop = g.location['x'] + 14
    ytop = g.location['y'] + 32

    xbot = g.location['x'] + g.size['width'] - 14
    ybot = g.location['y'] + g.size['height'] - 14

    im = Image.open('snake.png')
    im = im.crop((xtop, ytop, xbot, ybot))
    # im.save('snake.png')
    # -----------------------
    X = np.array([item[0] for item in im.getdata()]).reshape(im.height, im.width)
    X[X <= 100] = 1
    X[X > 100] = 0
    return X

def reset_game(browser, gameLevel):
    browser.find_element_by_id('buttonNew').click()
    browser.execute_script('gameLevel = ' + str(gameLevel))

def get_reward(browser, gameLevel, prevScore):
    """
    Return marginal reward from the previous time-step
    and the current score
    When dead, what the score should be?
    """
    dead = is_dead(browser)
    score_str = browser.find_element_by_id('score').text
    if dead or score_str=='':
        return -1, 0
        # return -(gameLevel+1), 0
    elif int(score_str) > prevScore:
        return 1, int(score_str)
        # return (gameLevel+1), int(score_str)
    else:
        return 0, prevScore

def is_dead(browser):
    return bool(browser.execute_script('return crash()'))
from PIL import Image
import pickle
import time
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import matplotlib.pyplot as plt ##########################
from helper import *

actions = [Keys.UP, Keys.RIGHT, Keys.DOWN, Keys.LEFT]
actions_ = ['UP', 'RIGHT', 'DOWN', 'LEFT']
# should I include do-nothing action?
game_url = 'http://helpfulsheep.com/snake/'
batch_size = 2
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
render = True
resume = False

if render:
    browser = webdriver.Chrome()
else:
    browser = webdriver.PhantomJS()

browser.get(game_url)
browser.set_window_size(800, 800)
# set browser size to prevent problems

gameLevel = 9
# interval between update in miliseconds, from the game's script.js
speed = [658, 478, 378, 298, 228, 178, 138, 108, 88, 68, 48, 38, 28, 18, 8];
# interval = (speed[gameLevel] - 20)/ 1000. # convert to seconds
# browser.find_element_by_xpath("//input[@name='gameLevel'][@value='"+str(gameLevel)+"']").click()

x = get_screen(browser).ravel()
# hyper parameters
# height, width = x.shape
D = len(x)        # input units
H = 300           # hidden units
K = len(actions)  # output units

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)
    model['W2'] = np.random.randn(K, H) / np.sqrt(H)

grad_buffer = {k : np.zeros_like(v) for k,v in model.items()}
rmsprop_cache = {k : np.zeros_like(v) for k,v in model.items()}

def softmax(x):
    m = max(x)
    return np.exp(x - m) / sum(np.exp(x - m))

def discount_rewards(r):
    r = np.array(r, dtype='float')
    discounted_r = np.zeros_like(r)
    for i in range(len(r)):
        reward = 0
        for j in range(0, len(r)-i):
            reward += r[i+j] * gamma**j
        discounted_r[i] = reward
    return discounted_r

def get_dlogp(probs, action_ix):
    # y = [1 if i==action_ix else 0 for i in range(len(actions))]
    dlogp = -probs
    dlogp[action_ix] = 1 - probs[action_ix]
    return dlogp

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0
    logp = np.dot(model['W2'], h)
    probs = softmax(logp)
    return probs, h

def policy_backward(eph, epdlogp):
    dW2 = np.dot(epdlogp.T, eph)
    dh = np.dot(epdlogp, model['W2'])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}


xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
prevScore = 0

# click newgame button
reset_game(browser, gameLevel)
body = browser.find_element_by_tag_name('body')
while True:
    x = get_screen(browser).ravel()
    probs, h = policy_forward(x)
    action_ix = np.random.choice(range(len(actions)), p=probs)
    action = actions[action_ix]

    xs.append(x)
    hs.append(h)
    dlogp = get_dlogp(probs, action_ix)
    dlogps.append(dlogp)

    body.send_keys(action)
    reward, prevScore = get_reward(browser, gameLevel, prevScore)
    print('%.2f %.2f %.2f %.2f' % tuple(probs),
           action_ix, actions_[action_ix])
    reward_sum += reward
    drs.append(reward)

    if is_dead(browser):
        episode_number += 1

        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [],[],[],[]

        discounted_epr = discount_rewards(epr)
        # # normalizing the rewards
        # discounted_epr -= np.mean(discounted_epr)
        # discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k]

        if episode_number % batch_size == 0:
            for k,v in model.items():
                ## normal
                # model[k] += learning_rate * grad_buffer[k]

                # rmsprop
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
            print('weights updated')
            print(model['W2'][0, 0])

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0:
            pickle.dump(model, open('save.p', 'wb'))

        reward_sum = 0
        reset_game(browser, gameLevel)
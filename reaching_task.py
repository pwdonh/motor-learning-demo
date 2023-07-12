from psychopy import visual, core
from psychopy.event import Mouse
from psychtoolbox import WaitSecs, GetSecs
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen_no', default=1)
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--num_training', type=int, default=6)
    parser.add_argument('--num_perturb', type=int, default=6)
    parser.add_argument('--num_washout', type=int, default=6)
    parser.add_argument('--theta', type=float, default=.5)
    parser.add_argument('--pdeviant', type=float, default=0.)
    # this will check whether device_name is a substring of the device_name from linux
    # it is sometimes called US-4x4HR: USB Audio (hw:0,0) or US-4x4HR: USB Audio (hw:1,0), etc.
    parser.add_argument('--device_name', type=str, default='US-4x4HR: USB Audio')
    # Return parser instead of parsed arguments
    # to be able to add more arguments for special cases
    return parser

def get_datetime_string():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S"), now.strftime("%Y%m%d")

def transform_position(pos, theta=.45):
    x, y = pos
    new_x = x*np.cos(theta) - y*np.sin(theta)
    new_y = x*np.sin(theta) + y*np.cos(theta)
    return (new_x, new_y)

parser = get_argument_parser()
args = parser.parse_args()

num_repeat = args.num_training+args.num_perturb+args.num_washout
squares = np.array([0,1,2,3]*num_repeat)
thetas = [0.]*4*args.num_training + [args.theta]*4*args.num_perturb + [0.]*4*args.num_perturb
deviant = np.zeros(squares.shape, dtype=int)
if args.pdeviant>0.:
    num_deviant = int(np.round(num_repeat*4*args.pdeviant))
    min_distance = 4
    do_continue = True
    while do_continue:
        indices_deviant = np.random.randint(args.num_training*4,num_repeat*4,num_deviant)
        if not np.any(np.diff(np.sort(indices_deviant))<min_distance):
            do_continue = False
    squares = np.array(squares)
    for index_deviant in np.sort(indices_deviant):
        squares[index_deviant:] = (squares[index_deviant:]+1)%4
        deviant[index_deviant] = 1
df = pd.DataFrame(dict(
    square = squares, theta = thetas, deviant=deviant
))

win = visual.Window(
    color=(-.3333,-.3333,-.3333), 
    screen = args.screen_no,
    fullscr=args.fullscreen
)

mouse = Mouse(win=win, visible=False)

WaitSecs(1.)

square_1 = visual.Rect(win, .2, .2, pos=(.5, .5), lineColor=(255, 255, 255), lineWidth=3, fillColor=None)
square_2 = visual.Rect(win, .2, .2, pos=(-.5, .5), lineColor=(255, 255, 255), lineWidth=3, fillColor=None)
square_3 = visual.Rect(win, .2, .2, pos=(.5, -.5), lineColor=(255, 255, 255), lineWidth=3, fillColor=None)
square_4 = visual.Rect(win, .2, .2, pos=(-.5, -.5), lineColor=(255, 255, 255), lineWidth=3, fillColor=None)
squares = [square_1, square_2, square_3, square_4]
square_mouse = visual.Rect(win, .2, .2, pos=(0., 0.), lineColor=(255, 255, 255), lineWidth=3, fillColor=None)
cursor = visual.Rect(win, .1, .1, pos=(0., 0.))

def prepare_screen(lineColor=(255, 255, 255)):
    square_mouse.setLineColor(lineColor)
    square_mouse.draw()
    for square in squares:
        square.setLineColor(lineColor)
        square.draw()

def is_in(target, mouse, dev=.05):
    x_mouse, y_mouse = mouse
    x_target, y_target = target
    x_true = (x_mouse<x_target+dev)&(x_mouse>x_target-dev)
    y_true = (y_mouse<y_target+dev)&(y_mouse>y_target-dev)
    return x_true&y_true

try:

    prepare_screen()
    win.flip() 
    WaitSecs(1.)

    start = GetSecs()
    times = []

    square_mouse.setFillColor((255,255,255))

    trial = 0
    current_theta = 0.
    current_color = (255,255,255)
    period = 'move_to_start'

    while trial<df.shape[0]:
        pos = mouse.getPos()
        pos = transform_position(pos, theta=current_theta)
        if period=='move_to_start':
            # Wait for mouse is in start position
            if square_mouse.contains(pos):
                times.append(GetSecs())
                # Fill the square for this trial
                target = squares[df.loc[trial,'square']]
                current_theta = df.loc[trial,'theta']
                if current_theta==0.:
                    current_color = (255, 255, 255)
                else:
                    current_color = (255, 0, 0)
                square_mouse.setFillColor(None)
                target.setFillColor((255,255,255))
                period = 'move_to_target'
        if period=='move_to_target':
            if target.contains(pos):
                times.append(GetSecs())
                target.setFillColor(None)
                square_mouse.setFillColor((255,255,255))
                trial += 1
                period = 'move_to_start'
        prepare_screen(current_color)
        cursor.setPos(pos)
        cursor.draw()
        win.flip()

    win.close()

    rts = np.diff(times)
    np.save('rts_'+get_datetime_string()[0], rts)
    rts = [rts[0]]+list(rts)
    avg = np.array(rts).reshape(-1,8).mean(1)
    avg2 = np.array(rts).reshape(-1,2).mean(1)
    if args.theta>0.:
        fig,ax = plt.subplots()
        ax.plot(avg, '-o')
        ax.set_xticks([args.num_training/2-.5, args.num_training+args.num_perturb/2-.5, args.num_training+args.num_perturb+args.num_washout/2-.5])
        ax.set_xticklabels(['Training', 'Perturbation', 'Washout'])
        lims = ax.get_ylim()
        ax.set_ylabel('Reaction time (sec)')
        ax.vlines(args.num_training-.5,lims[0],lims[1])
        ax.vlines(args.num_training+args.num_perturb-.5,lims[0],lims[1])
        fig.show()
    if args.pdeviant>0.:
        fig,ax = plt.subplots()
        ax.plot(avg2, 'o')
        ax.plot(np.where(df['deviant']==1)[0], avg2[df['deviant']==1],'o')
        ax.set_ylabel('Reaction time (sec)')
        ax.set_xlabel('Trial number')
        plt.legend(['Standard sequence', 'Deviant'])
        fig.show()

    input()

except KeyboardInterrupt:

    win.close()
    core.quit()

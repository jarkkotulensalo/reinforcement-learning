import sys,tty,os,termios
def getkey():
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            b = os.read(sys.stdin.fileno(), 3).decode()
            if len(b) == 3:
                k = ord(b[2])
            else:
                k = ord(b)
            key_mapping = {

                32: 0,

                65: 1,
                66: 2,

            }
            return key_mapping.get(k, chr(k))
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
try:
    while True:
        k = getkey()
        if k == 'esc':
            quit()
        else:
            print(k)
except (KeyboardInterrupt, SystemExit):
    os.system('stty sane')
    print('stopping.')

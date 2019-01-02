import logging

def logInit(fileName,resume=False):
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d-%H-%M-%S',
            filename=fileName,
            filemode= 'a' if resume else 'w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

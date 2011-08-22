# -*- coding: utf-8 -*-  

class bcolors:
    """
    Output színezésre szolgáló osztály
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    TESZT = '\033[90m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''


def print_cow(cowsay):
  """
  string printelése szövegbuborékban
  """
  lne0 = 2+len(cowsay)
  line0 = "   "
  line0 = line0+"_"*lne0
  lne = 2+len(cowsay)
  line = "-"*lne

  cowtxt= line0+"\n\
  < "+cowsay+" > \n\
   "+line+" \n\
         \   ^__^ \n\
          \  (oo)\_______ \n\
             (__)\       )\/\ \n\
                 ||----w | \n\
                 ||     ||\
                     "
  print cowtxt

from utils.consolecolors import bcolors

def info(str: str) -> None:
    print(f"{bcolors.OKCYAN}[INFO] {str}{bcolors.ENDC}")
    
def warn(str: str) -> None:
    print(f"{bcolors.WARNING}[WARNING] {str}{bcolors.ENDC}")
    

def header(str: str) -> None:
    print(f"{bcolors.HEADER}[INFO] {str}{bcolors.ENDC}")
   
def err(str: str) -> None:
    print(f"{bcolors.FAIL}[ERROR] {str}{bcolors.ENDC}")
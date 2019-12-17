def print_runtime(name, runtime):
    """
    Outputs the runtime to the terminal in seconds (with 2 decimals).
    :param name: The name of measured program.
    :param runtime: The runtime in seconds.
    :return: None
    """
    print("{} Runtime: {} seconds".format(name, runtime))


def print_ascii_title():
    """
    Ascii art used for the CLI.
    :return: None
    """
    print(""" _______ _      _        _      _                                 
|__   __(_)    | |      | |    | |                                
   | |   _  ___| | _____| |_   | |     ___   __ _  __ _  ___ _ __ 
   | |  | |/ __| |/ / _ \ __|  | |    / _ \ / _  |/ _  |/ _ \  __|
   | |  | | (__|   <  __/ |_   | |___| (_) | (_| | (_| |  __/ |   
   |_|  |_|\___|_|\_\___|\__|  |______\___/ \__, |\__, |\___|_|   
                                             __/ | __/ |          
                                            |___/ |___/          """)

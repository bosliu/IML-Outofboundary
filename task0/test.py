from turtle import left


class person:
    def __init__(self, leave=False, wave=0, hurry=0) -> None:
        self.leave = leave
        self.wave = wave
        self.hurry_up_MDFK = hurry


class hand:
    def __init__(self, carrying, wave=False) -> None:
        self.carrying = carrying
        self.wave = wave


class object:
    def __init__(self, left_hand: hand, right_hand: hand, charisma=0, melt=0) -> None:
        self.left_hand = left_hand
        self.right_hand = right_hand
        self.charisma = charisma
        self.melt = melt


def charisma_generator(zr: object, girl: person, dad: person):
    while not girl.leave:
        dad.hurry_up_MDFK += 1
        if zr.left_hand.wave and zr.left_hand.carrying == []:
            girl.leave = True
            print("ZR waved his left hand with ", zr.left_hand.carrying)
            print("Good bye accepted, the girl leaving")
            print("The girl waved her hands for ", girl.wave, "times")
            print("The Dad has said 'hurry up MDFK' for ",
                  dad.hurry_up_MDFK, "times")
            return zr.charisma+1

        zr.right_hand.carrying.extend(zr.left_hand.carrying)
        if zr.left_hand.wave:
            print("ZR waved his left hand with ", zr.left_hand.carrying)
        if zr.right_hand.wave:
            print("ZR waved his right hand with ", zr.right_hand.carrying)
        print("Good bye not accepted, switching hands...")
        zr.left_hand.carrying = []
        temp = zr.left_hand.wave
        zr.left_hand.wave = zr.right_hand.wave
        zr.right_hand.wave = temp
        girl.wave += 1


left_hand = hand(["milk"], True)
right_hand = hand(["debit"], False)
zr = object(left_hand, right_hand)
girl = person()
dad = person()
print("ZR's charisma is: ", charisma_generator(zr, girl, dad))

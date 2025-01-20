import multiprocessing as mp
import time

class mpGenSys:
    _dataList = []
    _genProcess = None
    lastTel = 0

    def __init__(self):
        print("initted gen")
        self.manager = mp.Manager()
        self._dataList = self.manager.list()  # Shared list for processes
        self._genProcess = mp.Process(target=self.startGen)
        self._genProcess.start()
        print("succesful")

    def startGen(self):
        counter = 0
        while True:  # Corrected the condition
            time.sleep(1) 
            self._dataList.append(len(self._dataList))
           

    def tellCount(self):
        if len(self._dataList) > self.lastTel:
            print("data added")
            self.lastTel+=1

test = mpGenSys()

while True:
    test.tellCount()


"""
Key Features of multiprocessing.Manager
Shared Objects: Manager allows you to create shared objects like lists, dictionaries, queues, etc., 
which can be accessed and modified by multiple processes.

Synchronization: It automatically handles the necessary synchronization so that multiple processes can 
safely access or modify shared objects without corrupting data.

Ease of Use: The API provided by Manager is intuitive and closely mirrors the corresponding built-in data 
structures, making it easy to integrate into programs.


"""
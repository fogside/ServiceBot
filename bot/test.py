import time
import asyncio

def coroutine(f):
    def wrap(*args,**kwargs):
        gen = f(*args,**kwargs)
        gen.send(None)
        return gen
    return wrap

class WaitInput(object):
    def __init__(self):
        self.coro = self.coro()

    @coroutine
    def coro(self):
        while True:
            message= (yield )
            if message is not None:
                yield message
                self.coro.close()


wait_input = WaitInput()

# async def test():
#     result = await wait_input
#     print('awaited. Result = {}'.format(result))
#     return result
#
# # c=test()
index = 0
while True:
    time.sleep(1)
    print('main')
    index+=1
    if index>=1:
        print('Awaited! Result = {}'.format(wait_input.coro.send(index)))
        wait_input = WaitInput()
#
# loop = asyncio.get_event_loop()
# loop.run_until_complete(test())
# loop.close()
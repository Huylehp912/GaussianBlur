import threading
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] (%(threadName)-10s) %(message)s',)

# ~ def worker(num):
	# ~ """thread worker function"""
	# ~ print 'Worker: %s' % num
	# ~ print "Worker: " + "{}".format(num)
	# ~ return

# ~ threads = []

# ~ for i in range(5):
	# ~ t = threading.Thread(target=worker, args = (i,))
	# ~ threads.append(t)
	# ~ t.start()
	
def worker():
	logging.debug('Starting')
	# ~ print threading.currentThread().getName(), 'Starting'
	time.sleep(2)
	# ~ print threading.currentThread().getName(), 'Exiting'
	logging.debug('Exiting')

def my_service():
	logging.debug('Starting')
	# ~ print threading.currentThread().getName(), 'Starting'
	time.sleep(3)
	# ~ print threading.currentThread().getName(), 'Exiting'
	logging.debug('Exiting')

main_thread = threading.currentThread()
logging.debug('hello %s', main_thread.getName())
t = threading.Thread(name='Worker-Huy', target=worker)
w = threading.Thread(name='Service-Lee', target=my_service)
w2 = threading.Thread(target=worker) # use default name

t.start()
w.start()
w2.start()

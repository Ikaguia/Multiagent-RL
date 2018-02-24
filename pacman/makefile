test-ghost-%:
	(python controller.py &)
	(python adapter.py --num-ghosts 2 --pacman-agent fleet -l 100 -t 10 --ghost-agent $* &)
test-ghost-%-4:
	(python controller.py &)
	(python adapter.py --num-ghosts 4 --pacman-agent fleet -l 100 -t 10 --ghost-agent $* &)
test-fghost-%:
	(python controller.py &)
	(python adapter.py --num-ghosts 2 --pacman-agent fleet -l 1 -t 100 --ghost-agent $* &)
test-fghost-%-4:
	(python controller.py &)
	(python adapter.py --num-ghosts 4 --pacman-agent fleet -l 1 -t 100 --ghost-agent $* &)
test-g-flee:   test-fghost-fixedFlee
test-g-seek:   test-fghost-fixedSeek
test-g-pursue: test-fghost-fixedPursue
test-g-ai2:    test-ghost-ai2
test-g-ai2-4:  test-ghost-ai2-4
stop:
	pkill python

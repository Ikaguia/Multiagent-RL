POLICIES_PATH = policies
RESULTS_PATH = results

ADAPTER	= python adapter.py
PM_FLEET= --pacman-agent fleet
GHOST	= --ghost-agent
NGHOSTS = --num-ghosts
TEST	= -l 1 -t 100
LEARN	= -l 100 -t 1
POLICY	= --policy-file


#gen policy files
$(POLICIES_PATH)/%_2g_fleet_100: stop controller
	@if ! test -f $(POLICIES_PATH)/$*; then\
		($(ADAPTER) $(NGHOSTS) 2 $(PM_FLEET) $(LEARN) $(GHOST) $* $(POLICY) $@);\
		(mv results.txt $(RESULTS_PATH)/learn-$*_2g_fleet_100);\
	fi
$(POLICIES_PATH)/%_3g_fleet_100: stop controller
	@if ! test -f $(POLICIES_PATH)/$*; then\
		($(ADAPTER) $(NGHOSTS) 3 $(PM_FLEET) $(LEARN) $(GHOST) $* $(POLICY) $@);\
		(mv results.txt $(RESULTS_PATH)/learn-$*_3g_fleet_100);\
	fi
$(POLICIES_PATH)/%_4g_fleet_100: stop controller
	@if ! test -f $(POLICIES_PATH)/$*; then\
		($(ADAPTER) $(NGHOSTS) 4 $(PM_FLEET) $(LEARN) $(GHOST) $* $(POLICY) $@);\
		(mv results.txt $(RESULTS_PATH)/learn-$*_4g_fleet_100);\
	fi
#gen new policy files
learn-ghost-%-2: old-$(POLICIES_PATH)/%_2g_fleet_100 $(POLICIES_PATH)/%_2g_fleet_100
	@true
learn-ghost-%-3: old-$(POLICIES_PATH)/%_3g_fleet_100 $(POLICIES_PATH)/%_3g_fleet_100
	@true
learn-ghost-%-4: old-$(POLICIES_PATH)/%_4g_fleet_100 $(POLICIES_PATH)/%_4g_fleet_100
	@true
#test learned policies
test-ghost-%-2: stop controller
	($(ADAPTER) $(NGHOSTS) 2 $(PM_FLEET) $(TEST) $(GHOST) $* $(POLICY) $(POLICIES_PATH)/$*_2g_fleet_100)
	(mv results.txt $(RESULTS_PATH)/test-$*_3g_fleet_100)
test-ghost-%-3: stop controller
	($(ADAPTER) $(NGHOSTS) 3 $(PM_FLEET) $(TEST) $(GHOST) $* $(POLICY) $(POLICIES_PATH)/$*_3g_fleet_100)
	(mv results.txt $(RESULTS_PATH)/test-$*_3g_fleet_100)
test-ghost-%-4: stop controller
	($(ADAPTER) $(NGHOSTS) 4 $(PM_FLEET) $(TEST) $(GHOST) $* $(POLICY) $(POLICIES_PATH)/$*_4g_fleet_100)
	(mv results.txt $(RESULTS_PATH)/test-$*_3g_fleet_100)


test-fghost-%: stop controller
	($(ADAPTER) $(NGHOSTS) 2 $(PM_FLEET) $(TEST) $(GHOST) $*)
test-fghost-%-4: stop controller
	($(ADAPTER) $(NGHOSTS) 4 $(PM_FLEET) $(TEST) $(GHOST) $*)
test-fghosts-fs: stop controller
	($(ADAPTER) $(NGHOSTS) 2 $(PM_FLEET) $(TEST) $(GHOST) fixedFlee $(GHOST)2 fixedSeek)
test-fghosts-fp: stop controller
	($(ADAPTER) $(NGHOSTS) 2 $(PM_FLEET) $(TEST) $(GHOST) fixedFlee $(GHOST)2 fixedPursue)
test-fghosts-sp: stop controller
	($(ADAPTER) $(NGHOSTS) 2 $(PM_FLEET) $(TEST) $(GHOST) fixedSeek $(GHOST)2 fixedPursue)
test-fghosts-fsp: stop controller
	($(ADAPTER) $(NGHOSTS) 3 $(PM_FLEET) $(TEST) $(GHOST) fixedFlee $(GHOST)2 fixedSeek $(GHOST)3 fixedPursue)
test-g-flee:   test-fghost-fixedFlee
test-g-seek:   test-fghost-fixedSeek
test-g-pursue: test-fghost-fixedPursue
test-g-ai2:    test-ghost-ai2
test-g-ai2-4:  test-ghost-ai2-4

controller:
	(python controller.py &)
stop:
	(pkill python || true)
old-% :
	@if test -f $*; then mv $* $*_old; fi
old-$(POLICIES_PATH)/% :
	@if test -f $(POLICIES_PATH)/$*; then mv $(POLICIES_PATH)/$* $(POLICIES_PATH)/$*_old; fi

.PHONY: old-$(POLICIES_PATH)/% old-% stop controller

POLICIES_PATH = policies
RESULTS_PATH = results

#constants to avoid repetition
ADAPTER	= python adapter.py
PACMAN  = --pacman-agent $(ARG_PMAGENT)
GHOST	= --ghost-agent
NGHOSTS = --num-ghosts $(ARG_NGHOSTS)
TEST	= -l 1 -t $(ARG_TGAMES)
LEARN	= -l $(LGAMES) -t 1
POLICY	= --policy-file $(POLICY_FILE)
SEED    = --seed $$($(ARG_SEED) * $(ARG_LSTEPS))
LGAMES  = $$(($(ARG_LSTEP) * $(ARG_LSTEPS)))

#default values, can be overwritten via command line
#i.e: 'make learn-ghost ARG_GHAGENT=ai ARG_NGHOSTS=2 ARG_LSTEPS=4'
#####
#number of test games
ARG_TGAMES = 10
#number of ghosts
ARG_NGHOSTS = 2
#each learn step is 50 learn games
ARG_LSTEP = 50
#2 learn steps, 100 learn games
ARG_LSTEPS = 2
#seed for random number generator
ARG_SEED = 123456789
#ghost agent
ARG_GHAGENT = ai
#pacman agent
ARG_PMAGENT = fleet
#clean
ARG_CLEAN = 1
####

ARGS_ = ARG_NGHOSTS=$(ARG_NGHOSTS) ARG_LSTEP=$(ARG_LSTEP) ARG_SEED=$(ARG_SEED) ARG_GHAGENT=$(ARG_GHAGENT) ARG_CLEAN=0
ARGS = $(ARGS_) ARG_LSTEPS=$(ARG_LSTEPS)

POLICY_FILE = $(POLICIES_PATH)/$(ARG_GHAGENT)_$(ARG_NGHOSTS)g_$(ARG_PMAGENT)_$(LGAMES)
L_RESULT_FILE = $(RESULTS_PATH)/learn-$(ARG_GHAGENT)_$(ARG_NGHOSTS)g_$(ARG_PMAGENT)_$(LGAMES).txt
T_RESULT_FILE = $(RESULTS_PATH)/test-$(ARG_GHAGENT)_$(ARG_NGHOSTS)g_$(ARG_PMAGENT)_$(LGAMES).txt
TMP_POLICY = $(POLICIES_PATH)/__temp


#rule to learn policies for ghosts
learn-ghost:
ifeq ($(ARG_CLEAN), 1)
	@(rm $(TMP_POLICY) || true)
endif
ifneq ($(shell test -e $(POLICY_FILE) && echo -n yes),yes)
	@echo "learning $(POLICY_FILE)"
ifneq ($(ARG_LSTEPS), 1)
	@($(MAKE) learn-ghost $(ARGS_) ARG_LSTEPS=$$(($(ARG_LSTEPS) - 1)))
endif
	($(MAKE) learn-policy $(ARGS))
else
	@echo "found $(POLICY_FILE)"
	@cp $(POLICY_FILE) $(TMP_POLICY)
endif
ifeq ($(ARG_CLEAN), 1)
	@(rm $(TMP_POLICY) || true)
endif

#rule to learn new policies for ghosts
relearn-ghost:
ifeq ($(ARG_CLEAN), 1)
	@(rm $(TMP_POLICY) || true)
endif
ifneq ($(ARG_LSTEPS), 1)
	@($(MAKE) relearn-ghost $(ARGS_) ARG_LSTEPS=$$(($(ARG_LSTEPS) - 1)))
endif
	@echo "($(MAKE) learn-policy $(ARGS))"
	@#($(MAKE) $(POLICY_FILE) $(ARGS))
ifeq ($(ARG_CLEAN), 1)
	@(rm $(TMP_POLICY) || true)
endif

#rule to make policy files
learn-policy: stop controller
	($(ADAPTER) $(SEED) -l $(ARG_LSTEP) -t 1 $(PACMAN) $(NGHOSTS) $(GHOST) $(ARG_GHAGENT) --policy-file $(TMP_POLICY))
	(cp $(TMP_POLICY) $(POLICY_FILE))
	(mv results.txt $(L_RESULT_FILE))
#test learned policies
test-ghost: stop controller
ifeq ($(shell test -e $(POLICY_FILE) && echo -n yes),yes)
	@(cp $(POLICY_FILE) $(TMP_POLICY))
	($(ADAPTER) $(SEED) $(TEST) $(PACMAN) $(NGHOSTS) $(GHOST) $(ARG_GHAGENT) $(POLICY) $(TMP_POLICY))
	(mv results.txt $(T_RESULT_FILE))
	@(rm $(TMP_POLICY))
else
	@(echo "Missing policy file '$(POLICY_FILE)'")
endif
#test fixed ghosts with no learning phase
test-fghost-%: stop controller
	($(ADAPTER) $(SEED) $(TEST) $(PACMAN) --num-ghosts 2 $(GHOST) $*)
test-fghosts-fs: stop controller
	($(ADAPTER) $(SEED) $(TEST) $(PACMAN) --num-ghosts 2 $(GHOST) fixedFlee $(GHOST)2 fixedSeek)
test-fghosts-fp: stop controller
	($(ADAPTER) $(SEED) $(TEST) $(PACMAN) --num-ghosts 2 $(GHOST) fixedFlee $(GHOST)2 fixedPursue)
test-fghosts-sp: stop controller
	($(ADAPTER) $(SEED) $(TEST) $(PACMAN) --num-ghosts 2 $(GHOST) fixedSeek $(GHOST)2 fixedPursue)
test-fghosts-fsp: stop controller
	($(ADAPTER) $(SEED) $(TEST) $(PACMAN) --num-ghosts 3 $(GHOST) fixedFlee $(GHOST)2 fixedSeek $(GHOST)3 fixedPursue)


controller:#start the controller
	(python controller.py --seed $(ARG_SEED) &)
stop:#stop all previous python processes
	(pkill python || true)
old-% : #rename a file to prevent overwriting it	#old(file)
	@#if file exists
ifeq ($(shell test -e $* && echo -n yes),yes)
	@#if file_old exists
ifeq ($(shell test -e $*_old && echo -n yes),yes)
	@#recursively call old(file_old)
	($(MAKE) old-$*_old)
endif
	@#rename file to file_old
	(mv $* $*_old)
endif
old-$(POLICIES_PATH)/% : #same as old-%
ifeq ($(shell test -e $(POLICIES_PATH)/$* && echo -n yes),yes)
ifeq ($(shell test -e $(POLICIES_PATH)/$*_old && echo -n yes),yes)
	($(MAKE) old-$(POLICIES_PATH)/$*_old)
endif
	(mv $(POLICIES_PATH)/$* $(POLICIES_PATH)/$*_old)
endif

debug:
	@echo $(POLICY_FILE)
test: stop controller
	($(ADAPTER) $(SEED) -l 1 -t 1 $(PACMAN) $(NGHOSTS) $(GHOST) $(ARG_GHAGENT))

.PHONY: old-$(POLICIES_PATH)/% old-% stop controller
.PHONY: test-fghosts-fsp test-fghosts-sp test-fghosts-fp test-fghosts-fs test-fghost-%
.PHONY: test-ghost-% learn-ghost-% $(POLICY_FILE)

.PRECIOUS: $(POLICIES_PATH)/%
.PRECIOUS: $(RESULTS_PATH)/%

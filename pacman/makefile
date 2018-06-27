SHELL=bash



POLICIES_PATH = policies
RESULTS_PATH = results

#constants to avoid repetition
ADAPTER	= python adapter.py
PACMAN  = --pacman-agent $(ARG_PMAGENT)
GHOST	= --ghost-agent
NGHOSTS = --num-ghosts $(ARG_NGHOSTS)
TEST	= -l 1 -t $(ARG_TGAMES)
LEARN	= -l $(LGAMES) -t 1
POLICY	= --policy-file
SEED    = --seed $$(($(ARG_SEED) * $(ARG_LSTEPS)))
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
#4 learn steps, 200 learn games
ARG_LSTEPS = 4
#seed for random number generator
ARG_SEED = 123456789
#ghost agent
ARG_GHAGENT = ai
#pacman agent
ARG_PMAGENT = fleet
####

ARGS = ARG_TGAMES=$(ARG_TGAMES) ARG_NGHOSTS=$(ARG_NGHOSTS) ARG_LSTEP=$(ARG_LSTEP) ARG_LSTEPS=$(ARG_LSTEPS) ARG_SEED=$(ARG_SEED) ARG_GHAGENT=$(ARG_GHAGENT) ARG_PMAGENT=$(ARG_PMAGENT)

POLICY_FILE = $(POLICIES_PATH)/$(ARG_GHAGENT)_$(ARG_NGHOSTS)g_$(ARG_PMAGENT)_$(LGAMES)
L_RESULT_FILE = $(RESULTS_PATH)/learn-$(ARG_GHAGENT)_$(ARG_NGHOSTS)g_$(ARG_PMAGENT)_$(LGAMES).txt
T_RESULT_FILE = $(RESULTS_PATH)/test-$(ARG_GHAGENT)_$(ARG_NGHOSTS)g_$(ARG_PMAGENT)_$(LGAMES).txt
TMP_POLICY = $(POLICIES_PATH)/__temp

all:
	@for(( i=1; i<=5; ++i ))do\
		a=ai;\
		if [[ $$i -ne 1 ]]; then a=$$a$$i; fi;\
		for(( j=2; j<=4; j++ ))do\
			echo $(MAKE) learn-ghost ARG_GHAGENT=$$a ARG_NGHOSTS=$$j ARG_LSTEP=100 ARG_LSTEPS=1;\
			$(MAKE) --no-print-directory learn-ghost ARG_GHAGENT=$$a ARG_NGHOSTS=$$j ARG_LSTEP=100 ARG_LSTEPS=1;\
			echo $(MAKE) test-ghost ARG_GHAGENT=$$a ARG_NGHOSTS=$$j ARG_LSTEP=100 ARG_LSTEPS=1;\
			$(MAKE) --no-print-directory test-ghost ARG_GHAGENT=$$a ARG_NGHOSTS=$$j ARG_LSTEP=100 ARG_LSTEPS=1;\
		done;\
	done

learn-all:
	@for(( i=1; i<=5; ++i ))do\
		a=ai;\
		if [[ $$i -ne 1 ]]; then a=$$a$$i; fi;\
		for(( j=2; j<=4; j++ ))do\
			echo $(MAKE) learn-ghost ARG_GHAGENT=$$a ARG_NGHOSTS=$$j ARG_LSTEP=100 ARG_LSTEPS=1;\
			$(MAKE) --no-print-directory learn-ghost ARG_GHAGENT=$$a ARG_NGHOSTS=$$j ARG_LSTEP=100 ARG_LSTEPS=1;\
		done;\
	done

test-all:
	@for(( i=1; i<=5; ++i ))do\
		a=ai;\
		if [[ $$i -ne 1 ]]; then a=$$a$$i; fi;\
		for(( j=2; j<=4; j++ ))do\
			echo $(MAKE) test-ghost ARG_GHAGENT=$$a ARG_NGHOSTS=$$j ARG_LSTEP=100 ARG_LSTEPS=1;\
			$(MAKE) --no-print-directory test-ghost ARG_GHAGENT=$$a ARG_NGHOSTS=$$j ARG_LSTEP=100 ARG_LSTEPS=1;\
		done;\
	done

#rule to learn policies for ghosts
learn-ghost:
	@clear || cls
	@rm -f $(TMP_POLICY)
	@rm -f results.txt
	@for(( i=1; i<=$(ARG_LSTEPS); ++i ))do\
		f=$(POLICIES_PATH)/$(ARG_GHAGENT)_$(ARG_NGHOSTS)g_$(ARG_PMAGENT)_$$((( $$i * $(ARG_LSTEP) )));\
		echo "#looking for $$f";\
		if [ -e $$f ]; then\
			echo "#found it";\
		else\
			echo "#didnt find it, building it instead";\
			if [[ ! $$i -eq 1 ]]; then\
				echo "(cp $$p $(TMP_POLICY))";\
				cp $$p $(TMP_POLICY);\
			fi;\
			$(MAKE) learn-policy $(ARGS) LGAMES=$$((( $$i * $(ARG_LSTEP) )));\
		fi;\
		p=$$f;\
	done
	@rm -f $(TMP_POLICY)
	@echo "#done"

#rule to learn new policies for ghosts
relearn-ghost:
	@clear || cls
	@rm -f $(TMP_POLICY)
	@rm -f results.txt
	for(( i=1; i<$(ARG_LSTEPS); ++i ))do\
		f=$(POLICIES_PATH)/$(ARG_GHAGENT)_$(ARG_NGHOSTS)g_$(ARG_PMAGENT)_$$((( $$i * $(ARG_LSTEP) )));\
		echo "#learning $$f";\
		if [[ ! $$i -eq 1 ]]; then\
			echo "(cp $$p $(TMP_POLICY))";\
		fi;\
		$(MAKE) learn-policy $(ARGS) LGAMES=$$((( $$i * $(ARG_LSTEP) )));\
		p=$$f;\
	done
	@rm -f $(TMP_POLICY)
	@echo "#done"

#rule to make policy files
learn-policy: stop controller
	($(ADAPTER) $(SEED) -l $(ARG_LSTEP) -t 1 $(PACMAN) $(NGHOSTS) $(GHOST) $(ARG_GHAGENT) $(POLICY) $(TMP_POLICY))
	(cp $(TMP_POLICY) $(POLICY_FILE))
	(mv results.txt $(L_RESULT_FILE))
#test learned policies
test-ghost: stop controller
	@if [ -e $(POLICY_FILE) ];then\
		if [ -e $(T_RESULT_FILE) ];then\
			cp $(POLICY_FILE) $(TMP_POLICY);\
			echo "($(ADAPTER) $(SEED) $(TEST) $(PACMAN) $(NGHOSTS) $(GHOST) $(ARG_GHAGENT) $(POLICY) $(TMP_POLICY))";\
			$(ADAPTER) $(SEED) $(TEST) $(PACMAN) $(NGHOSTS) $(GHOST) $(ARG_GHAGENT) $(POLICY) $(TMP_POLICY);\
			mv results.txt $(T_RESULT_FILE);\
			rm $(TMP_POLICY);\
		fi;\
	else\
		echo "Missing policy file '$(POLICY_FILE)'";\
	fi;
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
	@if [ -e $* ];then\
		if [ -e $*_old ];then\
			$(MAKE) old-$*_old;\
		fi;\
		mv $* $*_old;\
	fi;\
old-$(POLICIES_PATH)/% : #same as old-%
	@if [ -e $(POLICIES_PATH)/$* ];then\
		if [ -e $(POLICIES_PATH)/$*_old ];then\
			$(MAKE) old-$(POLICIES_PATH)/$*_old;\
		fi;\
		mv $(POLICIES_PATH)/$* $(POLICIES_PATH)/$*_old;\
	fi;\

debug:
	@echo $(POLICY_FILE)
test: stop controller
	($(ADAPTER) $(SEED) -l 1 -t 1 $(PACMAN) $(NGHOSTS) $(GHOST) $(ARG_GHAGENT))


.PHONY: old-$(POLICIES_PATH)/% old-% stop controller
.PHONY: test-fghosts-fsp test-fghosts-sp test-fghosts-fp test-fghosts-fs test-fghost-%
.PHONY: test-ghost-% learn-ghost-% $(POLICY_FILE)

.PRECIOUS: $(POLICIES_PATH)/%
.PRECIOUS: $(RESULTS_PATH)/%

DIRS = *_compute_performance load_performance

.PHONY:all clean $(DIRS)

all:$(DIRS)

clean:$(DIRS)

$(DIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)
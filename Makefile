# find all Makefile ./src/*/Makefile
PROJECTS=$(shell find src/ -name Makefile -type f -printf '%h\n' | sort -u)

all:
	@for project in $(PROJECTS); do \
		$(MAKE) --no-print-directory -C $$project; \
	done

clean:
	@for project in $(PROJECTS); do \
		$(MAKE) --no-print-directory -C $$project clean; \
	done
CX=clang++
C=clang
LD=-fuse-ld=mold
CFLAGS=--std=c++20 -Og -g -Wall -Wextra -Wno-format -Wno-write-strings

PREREQ_DIR=@mkdir -p $(@D)

SRCDIR=src
BUILDDIR=build
BINDIR=bin

NAME=$(addprefix $(BINDIR)/, mouse_logger)

SRCS=$(wildcard $(SRCDIR)/*.cc)#$(wildcard $(SRCDIR)/*/*.cc)
OBJS=$(patsubst $(SRCDIR)/%.cc, $(BUILDDIR)/%.o, $(SRCS))

all:
	@$(MAKE) --no-print-directory $(NAME)

re: clean
	@$(MAKE) --no-print-directory

$(NAME): $(OBJS) | $(@D)
	$(PREREQ_DIR)
	$(CXX) $(CFLAGS) -o $(NAME) $(OBJS) $(LFLAGS)

$(OBJS): $(BUILDDIR)/%.o: $(SRCDIR)/%.cc
	$(PREREQ_DIR)
	$(CXX) $(CFLAGS) -o $@ -c $<

clean:
	@echo
	@echo "cleaning old build files"
	@echo
	@$(MAKE) --no-print-directory cleanbuild
	@$(MAKE) --no-print-directory cleanexec

cleanbuild:
	rm -f build/*.o

cleanexec:
	rm -f $(NAME)

.PHONY: all clean

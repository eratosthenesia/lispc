(headers
	stdio
	stdlib
	"string"
	pwd
	grp
	dirent
	time
	sys/types
	sys/stat)


(syn true "True")
(syn false "False")
(define True 1)
(define False 0)

(define (macro !execinfop !x)
	(.and !x (.or !s-ixusr !s-ixgrp !s-ixoth)))
(define (macro !chifelse !p !a !b)
	(@putchar (? !p (paren !a) !b)))
(define (macro !buffer !x)
	(? !x True False))
(define (macro !nop !x) !x)

(typedef (struct dirent) dirent)

(template incptr (x) (set x (addr (nth x 1))))
(func (pt parse-arg) char (

	((pt argv 2) char)
	((pt search 2) char)
	(argp char)
	((pt slot) int))

	(var i int 0)
	(while (ptr argv)
		(var (pt start 2) char search)
		(while (ptr start)
			(if (not (@strcmp (pt argv) (pt start)))
				(progn
					(if argp
						(return (nth argv 1)))
					(if (lt (pt slot) i)
						(set (pt slot) i))
					(return (str !true))))
			(incptr start))
		(incptr argv)
		(pinc i))
	(return !null))

(const (arr (pt !classify) nil)       char (arr-decl (str "-c") (str "-classify")        !null))
(const (arr (pt !diskusage) nil)      char (arr-decl (str "-d") (str "-disk-usage")      !null))
(const (arr (pt !longlisting) nil)    char (arr-decl (str "-l") (str "-long-listing")    !null))
(const (arr (pt !followsymlinks) nil) char (arr-decl (str "-f") (str "-follow-symlinks") !null))
(const (arr (pt !humanreadable) nil)  char (arr-decl (str "-h") (str "-human-readable")  !null))
(const (arr (pt !recursive) nil)      char (arr-decl (str "-r") (str "-recursive")       !null))


(func atoi-if int (((pt c) char))
	(if (not c)
		(return -1))
	(return (@atoi c)))

(define !bufsize 256)
(var (arr buffer !bufsize) char)

(func modeify void (x)
	(var (arr ret 10) char (str "drwxrwxrwx"))
	(var i int 9)
	(if (not (call !s-isdir x))
		(set (nth ret 0) (char "-")))
	(while (ge i 1)
		(if (same (mod x 2) 0)
			(set (nth ret i) (char "-")))
		(div-eq x 2)
		(pdec i))
	(for (set i 0) (lt i 10) (inc i)
		(@putchar (nth ret i)))
	(@putchar (char "\\t")))


(func printhumansize int (x)
	(var size float (^x float))
	(var (pt (arr prefixes nil)) char
		(arr-decl (str !b) (str !kb) (str !mb) (str !tb) (str !pb)))
	(var i int 0)
	(while (gt size 1024)
		(div-eq size 1024)
		(pinc i))
	(@printf (str "%.2g%s\\t") size (nth prefixes i)))

(func (pt stradd) char (((const (pt a) char) nil) ((const (pt b) char) nil))
	(var (pt c) char (@malloc (add (@strlen a) (@strlen b) 1)))
	(@memcpy c a (@strlen a))
	(@memcpy (add c (@strlen a)) b (add (@strlen b) 1))
	(return c))

(typedef (struct -ll (
	((pt next) (struct -ll))
	((pt data) void))) ll)

(func lsdir void (
	((pt path) char)
	(classify char)
	(longlisting char)
	(followsymlinks char)
	(humanreadable char)
	(recursive char))

	(var (pt dir) !dir (@opendir path))
	(if (not dir) (return))

	(var dirstat (struct stat))
	(var (pt dirinfo) dirent)
	(var (pt nextlistings) ll)
	(var (pt tmp) ll)

	(while (set dirinfo (@readdir dir))
		(@stat (slot dirinfo d-name) (addr dirstat))
		(@printf (str "%-12s\\t") (slot dirinfo d-name))
		(if recursive
			(if (and (eq (slot dirinfo d-type) !dt-dir)
				(or (eq (slot dirinfo d-type) !dt-lnk) followsymlinks))
				(if (and
					(@strcmp (slot dirinfo d-name) (str "."))
					(@strcmp (slot dirinfo d-name) (str "..")))
					(progn
						(set tmp nextlistings)
						(set nextlistings (@malloc (@sizeof ll)))
						(set (slot nextlistings data)
									(@stradd (@stradd path (str /)) (slot dirinfo d-name)))
						(set (slot nextlistings next) tmp)))))

		(if classify
			(progn
				(@!chifelse (eq (slot dirinfo d-type) !dt-dir)
					(char /) (char " "))
				(@!chifelse (@!execinfop (mem dirstat st-mode))
					(char *) (char " "))
				(@!chifelse (eq (slot dirinfo d-type) !dt-lnk)
					(char @) (char " "))))
		(if longlisting
			(progn
				(@printf (str %d\\t) (slot dirinfo d-ino))
				(@modeify (mem dirstat st-mode))
				(var (pt uid) (struct passwd) (@getpwuid (mem dirstat st-uid)))
				(var (pt gid) (struct group)  (@getgrgid (mem dirstat st-uid)))
				(@printf (str %s\\t) (? uid (slot uid pw-name) (str root)))
				(@printf (str %s\\t) (? gid (slot gid gr-name) (str root)))
				(if humanreadable
					(@printhumansize (mem dirstat st-size))
					(@printf (str %d\\t) (mem dirstat st-size)))


				(var (pt yearboundary) (struct tm))
				(var rawtime time_t (@time 0))
				(set yearboundary (@localtime (addr rawtime)))
				(set (slot yearboundary tm-yday) 0)
				(set (slot yearboundary tm-mon ) 0)
				(set (slot yearboundary tm-mday) 0)
				(set (slot yearboundary tm-sec ) 0)
				(set (slot yearboundary tm-min ) 0)
				(set (slot yearboundary tm-hour) 0)
				(if (lt (mem dirstat st-mtime) (@mktime yearboundary))
					(@strftime
						buffer
						!bufsize
						(str "%b %d %Y")
						(@localtime (addr (mem dirstat st-mtime))))
					(@strftime
						buffer
						!bufsize
						(str "%b %d %H:%M")
						(@localtime (addr (mem dirstat st-mtime)))))
				(@printf (str "%s ") buffer)))
		(@putchar (char \\n))
		(while nextlistings
			(@printf (str "\\n%s:\\n")
				(cast (slot nextlistings data) (typ* char 2)))
			(@lsdir
				(cast (slot nextlistings data) (typ* char 2))
				classify longlisting followsymlinks humanreadable recursive)
			(set nextlistings (slot nextlistings next)))))
(main
	(var pos int 0)
	(var diskusage int -1)
	(set diskusage
		(@atoi-if
			(@parse-arg argv (^!diskusage (typ* char 2)) true (addr pos))))
	(templates buffers-set (x y)
		(var x char (@!buffer (@parse-arg argv (cast y (typ* char 2)) false (addr pos)))))
	(buffers-set
		(classify !classify)
		(longlisting !longlisting)
		(followsymlinks !followsymlinks)
		(humanreadable !humanreadable)
		(recursive !recursive))
	(var (pt path) char (? (eq pos (- argc 1)) (str ".") (nth argv (- argc 1))))
	(@lsdir path classify longlisting followsymlinks humanreadable recursive)
	(return 0))



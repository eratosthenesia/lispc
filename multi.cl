(headers stdio stdlib string
		 pthread)
(template loop-n (n v maxv body)
	(block
		(
			(var v)
			(var maxv int n)
			(for (= v 0) (< v maxv) (++ v) body))))

(template voidint (x) (cast x size-t (typ* void)))
(template intvoid (x) (cast x size-t int))

(func (pt threadfunc) void (((pt x) void))
	(@sleep 0)
	(var i int (intvoid x))
	(@printf (str "Hello from thread %d\\n") (+ 1 (intvoid x)))
	(return !null))

(define !nthreads 12)

(main
	(var (arr threads !nthreads) pthread-t)
	(loop-n !nthreads i maxi
		(@PTHREAD-CREATE (addr (nth threads i)) !null (addr threadfunc) (voidint i)))
	(loop-n !nthreads i maxi
		(@PTHREAD_JOIN (nth threads i) !null))
	(return 0))
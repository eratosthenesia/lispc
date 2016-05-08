(headers
    ("../common/book.h" :local t)
    ("../common/cpu_bitmap.h" :local t))

(template sq (x) (* x x))
(define !dim 1000)

(struct cu-complex (
    (r real)
    (i real)))

(template cu-complex-decl (rv iv)
    (cast (arr-decl rv iv) (struct cu-complex)))

(func cu-complex-magnitude float ((x cu-complex))
    (return (+ (sq (.> x i)) (sq (.> x r)))))

(func cu-complex-mul cu-complex ((x cu-complex) (y cu-complex))
    (var z cu-complex)
    (= (.> z r) (-
                    (* (.> x r) (.> y r))
                    (* (.> x i) (.> y i))))
    (= (.> z i) (+
                    (* (.> x i) (.> y r))
                    (* (.> x r) (.> y i))))
    (return z))

(func cu-complex-add cu-complex ((x cu-complex) (y cu-complex))
    (var z cu-complex)
    (= (.> z r) (+ (.> x r) (.> y r)))
    (= (.> z i) (+ (.> x i) (.> y i)))
    (return z))

(cuda/device julia int (x y)
    (const scale float 1.5)
    (template jvar (v)
        (var (sym/add j v) float
            (* scale (cast (/ (- (/ !dim 2) x) (/ !dim 2))))))
    (jvar x) (jvar y)
    (template cvar (v x y)
        (var v (struct cu-complex) (cu-complex-decl x y)))
    (cvar c -9.8 0.156) (cvar a jx jy)
    (var i int 0)
    (for (= i 0) (< i 200) (++ i)
        (= a (@cu-complex-add (@cu-complex-mul a a) c))
        (if (> (@cu-complex-magnitude a) 1000) (return 0)))
    (return 1)
)

(cuda/global kernel void (((pt ptr) char nil unsigned))
    (var x int block/idx/x)
    (var y int block/idx/y)
    (var offset int (+ x (* y grid/dim/x)))
    (var julia-value int (@julia x y))
    (= ([]ptr (+ 0 (* offset 4))) (* 255 julia-value))
    (= ([]ptr (+ 1 (* offset 4))) 0)
    (= ([]ptr (+ 2 (* offset 4))) 0)
    (= ([]ptr (+ 3 (* offset 4))) 255))

(syn cpubitmap "CPUbitmap")

(main
    (var (@bitmap !dim !dim) cpubitmap)
    (var (pt dev-bitmap) char nil unsigned) 
    (@!handle-error
        (@cuda/malloc
            (cast (addr dev-bitmap) (typ* void 2))
            (.> bitmap (@image-size))))
    (var (@grid !dim !dim) dim3)
    (cuda/call kernel (grid 1) dev-bitmap)
    (@!handle-error
        (@cuda/memcpy
            (.> bitmap (@get-ptr))
            dev-bitmap
            (.> bitmap (@image-size))
            cuda/dev->host))
    (.> bitmap (@display-and-exit))
    (@!handle-error
        (@cuda/free dev-bitmap)))
; Copyright Jonathan Baca, 2016

(defparameter *file-out* nil)
(defparameter *exec-out* nil)
(defparameter *last-compiled* nil)
(defparameter *c-synonyms* (make-hash-table))

(defmacro sethash (k v hash)
  `(setf (gethash ,k ,hash) ,v))

(defmacro inhash (k hash)
  `(nth-value 1 (gethash ,k ,hash)))

(defmacro csyn (k v)
  `(sethash ,k ,v *c-synonyms*))

(defmacro cunsyn (k)
  `(remhash ,k *c-synonyms*))

(defun write-out (str)
  (if *file-out*
      (with-open-file (stream *file-out* :direction :output :if-exists :append :if-does-not-exist :create)
                      (format stream str))))

(defun change-file (file &optional is-h)
  (setf *exec-out* (c-strify file))
  (setf *file-out* (format nil "~a.~a" *exec-out* (if is-h #\h #\c))))

(defun change-exec (nym)
  (setf *exec-out* (c-strify nym)))

(defun compile-c ()
  (ext:run-shell-command (format nil "gcc ~a -o ~a" *file-out* *exec-out*)))

(defun strof (x)
    (format nil "~a" x))

(defun f/list (x)
    (if (listp x) x (list x)))

(defun f/list/n (x &optional (n 1))
  (if (zerop n) x
      (if (eq 1 n) (f/list x)
          (mapcar #'(lambda (y) (f/list/n y (1- n))) (f/list x)))))

(defun strsof (xs)
    (format nil "~{~a~}" xs))

(defun chs->str (x)
    (strsof x))

(defun str->chs (x)
    (loop for c across x collect c))

(defun replace-char (before after str)
    (chs->str (mapcar #'(lambda (x) (if (eq x before) after x)) (str->chs str))))

(defun numeric-string (x)
  (ignore-errors (numberp (read-from-string x))))

(defun c-strify (x)
    (if (stringp x) x
        (let ((s (strof x)))
          (if (numeric-string s) s
              (if (eq (char s 0) #\!) (subseq s 1)
                  (replace-char #\- #\_ (string-downcase s)))))))

(defmacro sethash (k v hash)
    `(setf (gethash ,k ,hash) ,v))

(defun addsyms (&rest syms)
    (read-from-string (strsof syms)))

(defun macn (x &optional n)
    (def n 1)
    (if (zerop n) x (macn (macroexpand-1 x) (1- n))))

(defmacro def (a b) `(setf ,a (if ,a ,a ,b)))

(defmacro deff (x f)
  `(setf ,x (,f ,x)))

(defmacro func-syn (func syn)
  `(progn
    (defun ,syn (&rest args)
      (apply #',func args))
    (compile ',syn)))

(defmacro cfunc-syn (func syn)
  `(func-syn ,(cnym func) ,(cnym syn)))

(defmacro func-syns (func syns &rest yns)
  (deff syns f/list)
  (setf syns (append syns yns))
  `(progn ,@(mapcar #'(lambda (syn) `(func-syn ,func ,syn)) syns)))

(defmacro cfunc-syns (func syns &rest yns)
  (deff syns f/list)
  (setf syns (append syns yns))
  `(progn ,@(mapcar #'(lambda (syn) `(cfunc-syn ,func ,syn)) syns)))

(defmacro un (x) `(setf ,x (not ,x)))

(defun cnym (nym)
  (nth-value 0 (addsyms nym '-c)))

(defmacro incr (x)
  `(setf ,x (1+ ,x)))

(defmacro cdefun (f args &body body)
  `(progn
     (defun ,(cnym f) ,args ,@body) 
     (compile ',(cnym f))))

(defmacro binop2 (oper &key nlp nrp nym)
    (def nym oper)
    (un nlp)
    (un nrp)
    (labels ((helper (a b) (if a `(format nil "(~a)" (cof ,b)) `(cof ,b))))
            `(cdefun ,nym (x y) (format nil "~a~a~a" ,(helper nlp 'x) ',oper ,(helper nrp 'y)))))

; does a left reduce
(defmacro lredop (oper &key nym nparen)
  (def nym oper)
 (let ((lp (if nparen "" "(")) (rp (if nparen "" ")")))
  `(cdefun ,nym (&rest xs)
     (if (null xs) nil
         (if (= 1 (length xs))
             (format nil "~a~a~a" ,lp (cof (car xs)) ,rp)
             (format nil "~a~a~a~a~a~a~a" ,lp ,lp (cof (car xs)) ,rp ',oper (apply (function ,(cnym nym)) (cdr xs)) ,rp))))))

(defmacro rredop (oper &key nym nparen)
  (def nym oper)
 (let ((lp (if nparen "" "(")) (rp (if nparen "" ")")))
    `(cdefun ,nym (&rest xs)
     (if (null xs) nil
         (if (= 1 (length xs))
             (format nil "~a~a~a" ,lp (cof (car xs)) ,rp)
             (format nil "~a~a~a~a~a~a~a" ,lp (apply (function ,(cnym nym)) (butlast xs)) ',oper ,lp (cof (car (last xs))) ,rp ,rp))))))


(defmacro binop (oper &key nlp nrp nym nyms l r nparen)
  ;;; (format t "OPER:~a NYM:~a NYMS:~a NPAREN:~a~%" oper nym nyms nparen)
    (if nyms
        `(progn ,@(mapcar #'(lambda (x) `(binop ,oper :nlp ,(un nlp) :nrp ,(un nrp) :nym ,x :l l :r r :nparen ,nparen)) nyms))
        (if (or l r)
            (if l `(lredop ,oper :nym ,nym :nparen ,nparen) `(rredop ,oper :nym ,nym :nparen ,nparen))
            `(binop2 ,oper :nlp ,nlp :nrp ,nrp :nym ,nym))))

(defmacro pre (oper &key nym nparen)
  `(cdefun ,nym (x) (format nil "~a~a~a~a" ',oper ,(if nparen "" "(") (cof x) ,(if nparen "" ")") )))
(defmacro post (oper &key nym nparen)
  `(cdefun ,nym (x) (format nil "~a~a~a~a" ,(if nparen "" "(") (cof x) ,(if nparen "" ")") ',oper)))
(defmacro prepost (oper &key post nym nparen nyms)
    (setf nym (if nym nym oper))
    (if nyms
        `(progn ,@(mapcar #'(lambda (x) `(prepost ,oper :post ,post :nym ,x :nparen ,nparen)) nyms))
        (if post
            `(post ,oper :nym ,nym :nparen ,nparen)
            `(pre ,oper :nym ,nym :nparen ,nparen))))
(defmacro preposts (&rest opers)
    `(progn ,@(mapcar #'(lambda (oper) `(prepost ,@(f/list oper))) opers)))
(defmacro binops (&rest opers)
    `(progn ,@(mapcar #'(lambda (oper) `(binop ,@(f/list oper))) opers)))

(defmacro swap (a b)
  (let ((c (gensym)))
  `(let ((,c ,a))
     (setf ,a ,b)
     (setf ,b ,c)
     (setf ,c ,a))))

(defmacro cfun (nym llisp &body body)
    `(cdefun ,nym ,llisp ,@body))

(defmacro cfuns (&body defs)
    `(progn ,@(mapcar #'(lambda (def) `(cfun ,@def)) defs)))

(defun c (&rest xs)
  (format nil "~{~a~^~(;~%~%~)~}" (mapcar #'cof xs)))

(defmacro cwrite (&rest xs)
  `(write-out (format nil "~a;~%" (c ,@xs))))

(defun symtrim (x n)
  (read-from-string (subseq (strof x) n)))

(defun cof (x)
  (if (null x) ""
      (if (atom x)
        (if (inhash x *c-synonyms*)
          (cof (gethash x *c-synonyms*))
          (c-strify x))
          (if (atom (car x))
              (case (char (strof (car x)) 0)
                  (#\@ (apply #'call-c (cof (symtrim (car x) 1)) (cdr x)))
                  (#\[ (apply #'nth-c (cof (symtrim (car x) 2)) (cdr x)))
                  (#\% (apply #'addr-c (cof (symtrim (car x) 1)) (cdr x)))
                  (#\^ (apply #'cast-c (cof (symtrim (car x) 1)) (cdr x)))
                  (#\$ (apply #'ptr-c (cof (symtrim (car x) 1)) (cdr x)))
                  (otherwise (apply (addsyms (car x) '-c) (cdr x))))
              (format nil "~{~a~^~(;~%~)~}" (mapcar #'cof x))))))

(defmacro cofy (x) `(setf ,x (cof ,x)))
(defmacro cofsy (x) `(setf ,x (mapcar #'cof (f/list ,x))))

(defun replacify (vars subs template)
    (labels ((helper (v s temp)
      (if (eq temp v) s
          (if (atom temp) temp
              (mapcar #'(lambda (x) (helper v s x)) temp)))))
            (if (null vars) template
                (replacify (cdr vars) (cdr subs) (helper (car vars) (car subs) template)))))

(defmacro replacify-lambda (vars template)
  (let ((varlist (loop for i from 1 to (length vars) collect (gensym))))
    `(lambda ,varlist (replacify ',vars (list ,@varlist) ',template))))

; ## NOW DEFINE THE C LANGUAGE

(binops (=   :l t :nyms (= set let <- ":="))
        (!=  :l t :nyms (!= neq diff different))
        (==  :r t :nyms (== eq same))
        (<   :r t :nyms (< lt))
        (>   :r t :nyms (> gt))
        (<=  :r t :nyms (<= leq))
        (>=  :r t :nyms (>= geq))
        (&&  :r t :nyms (&& and et und y))
        (&   :r t :nyms (& bit-and band .and bit-et bet .et bit-und bund .und bit-y by .y ))
        (&=  :l t :nyms (&= &-eq bit-and-eq band-eq .and-eq bit-et-eq bet-eq .et-eq bit-und-eq bund-eq
                              .und-eq bit-y-eq by-eq .y-eq &= bit-and= band= .and= bit-et= bet=
                              .et= bit-und= bund= .und= bit-y= by= .y= ))
        ("||":r t :nyms (or uel oder o))
        ("|" :r t :nyms (bit-or .or bor bit-uel .uel buel bit-oder .oder boder bit-o .o bo))
        ("|=":l t :nyms (bit-or-eq .or-eq bor-eq bit-uel-eq .uel-eq buel-eq bit-oder-eq
                                   .oder-eq boder-eq bit-o-eq .o-eq bo-eq bit-or= .or=
                                   bor= bit-uel= .uel= buel= bit-oder= .oder= boder= bit-o= .o= bo=))
        (+   :r t :nyms (+ plus add sum))
        (+=  :l t :nyms (+= plus-eq add-eq sum-eq plus= add= sum=))
        (-   :r t :nyms (- minus subtract sub))
        (-=  :l t :nyms (-= minus-eq subtract-eq sub-eq minus= subtract= sub=))
        (*   :r t :nyms (* times product mul multiply))
        (*=  :l t :nyms (*= times-eq product-eq mul-eq multiply-eq times= product= mul= multiply=))
        (/   :r t :nyms (/ quotient ratio div divide))
        (/=  :l t :nyms (/= quotient-eq ratio-eq div-eq divide-eq quotient= ratio= div= divide=))
        (%   :r t :nyms (% modulo mod remainder))
        (%=  :l t :nyms (%-eq modulo-eq mod-eq remainder-eq %= modulo= mod= remainder=))
        (<<  :r t :nyms (<< l-shift shift-left shl))
        (<<= :l t :nyms (<<= l-shift-eq shift-left-eq shl-eq l-shift= shift-left= shl=))
        (>>  :r t :nyms (>> r-shift shift-right shr))
        (>>= :l t :nyms (>>= r-shift-eq shift-right-eq shr-eq >>= r-shift= shift-right= shr=))
        (->  :nparen t :nyms (-> slot))
        (#\. :nrp t :nym mem))

(preposts (++ :post nil :nyms (++  inc +inc incr pre++ +1 ++n))
          (++ :post t   :nyms (+++ pinc inc+ pincr post++ 1+ n++))
          (-- :post nil :nyms (--  dec -dec decr pre-- -1 --n))
          (-- :post t   :nyms (--- pdec dec- pdecr post-- 1- n--))
          (-  :post nil :nyms (neg))
          (&  :post nil :nyms (addr memloc loc))
          (!  :post nil :nyms (! not un a flip))
          (~  :post nil :nyms (~ bit-not bit-un bit-a bit-flip))
          (*  :post t   :nyms (ptrtyp arg*) :nparen t))
          
(cfuns
  (arr-decl (&rest xs)
    (format nil "{~{~a~^~(, ~)~}}" (mapcar #'cof xs)))
  (sym/add (&rest xs)
    (cofsy xs)
    (strsof xs))
  (typ* (x &optional (n 1))
    (cofy x)
    (format nil "~a~{~a~}" x (loop for i from 1 to n collect #\*)))
  (const (&rest xs)
    (format nil "const ~a;~%" (apply #'var-c xs)))
  (syn (a b)
    (progn
      (csyn a b) ""))
  (unsyn (a)
    (progn
      (cunsyn a) ""))
  (progn (&rest xs)
         (format nil "~{  ~a;~^~%~}" (mapcar #'cof xs)))
  (? (test ifyes ifno)
    (cofy test)
    (cofy ifyes)
    (cofy ifno)
    (format nil "(~a)?~a:(~a)" test ifyes ifno))
  (if (test ifyes &optional ifno)
      (cofy test)
      (cofy ifyes)
      (cofy ifno)
      (format nil "if(~a){~%   ~a;~%}else{~%   ~a;~%}" test ifyes ifno))
  (cond (pairs)
        (format nil "if(~a){~{~%  ~a;~}~%}~{~a~}" (cof (caar pairs)) (mapcar #'cof (f/list (cadar pairs)))
                (mapcar #'(lambda (pair) (format nil "else if(~a){~{~%   ~a;~}~%}"
                                                 (cof (car pair)) (mapcar #'cof (f/list (cadr pair))))) (cdr pairs))))
  (main (&rest body)
        (format nil "int main(int argc,char** argv)~a" (block-c body)))
  (for (a b c &rest lines)
       (cofy a) (cofy b) (cofy c)
       (format nil "for(~a;~a;~a)~a" a b c (block-c lines)))
  (while (test &rest lines)
    (cofy test)
    (format nil "while(~a)~a" test (block-c lines)))
  (do-while (test &rest lines)
    (cofy test)
    (format nil "do~awhile(~a)" (block-c lines) test))
  (switch (var &rest pairs)
    (cofy var)
    (labels ((helper (pairs)
                (format nil "~a:~%   ~a~a;~%~a"
                  (cof (caar pairs))
                  (block-c (f/list (cadar pairs)))
                  (if (cddar pairs) (format nil "~%   break") "")
                  (if (cdr pairs)
                    (helper (cdr pairs))
                    ""))))
    (format nil "switch(~a){~a}" var (helper pairs))))
  (ptr (x &optional (n 1))
        (format nil "~{~a~}(~a)" (loop for i from 1 to n collect #\*) (cof x)))
  (pt (x &optional (n 1))
        (format nil "~{~a~}~a" (loop for i from 1 to n collect #\*) (cof x)))
  (nth (x &optional (n 0) &rest ns)
       (format nil "(~a)[~a]~a" (cof x) (cof n)
        (if ns
          (format nil "~{[~a]~}" (mapcar #'cof ns)) "")))
  (arr (x &optional (n 2) &rest ns)
        (format nil "~a[~a]~a" (cof x) (cof n)
          (if ns
          (format nil "~{[~a]~}" (mapcar #'cof ns)) "")))
  (call (nym &rest args)
    (format nil "~a(~{~a~^,~})" (cof nym) (mapcar #'cof args)))
  (cuda/call (nym ijk &rest args)
    (cofy nym) (cofsy ijk)
    (format nil "~a<<<~{~a~^,~}>>>(~{~a~^,~})" nym ijk (mapcar #'cof args)))
  (str (&rest x)
       (cofsy x)
       (format nil "\"~{~a~}\"" x))
  (cast (nym &optional (typ 'int) &rest typs)
    (if typs 
      (apply #'cast-c (cast-c nym typ) typs)
    (format nil "((~a)(~a))" (cof typ) (cof nym))))
  (var (x &optional (type 'int) init)
       (cofy x)
       (cofy type)
       (format nil "~a ~{~a~^,~}~a" type (f/list x) (if init (format nil "=~a" (cof init)) "")))
  (vars (x &optional (inter #\,))
        (setf x (mapcar #'(lambda (y) (apply #'var-c (f/list y))) (f/list/n x 1)))
        (format nil (format nil "~~{~~a~~^~(~a~%~)~~}" inter) x))
  (struct (nym &optional vars)
          (cofy nym)
          (if vars
              (format nil "struct ~a{~a;}" nym (vars-c vars #\;))
              (format nil "struct ~a" nym)))
  (block (lines)
         (format nil "{~%~{   ~a~^~(;~%~)~};~%}" (mapcar #'cof (f/list lines))))
  (func (nym typ vars &rest body)
        (cofy nym)
        (cofy typ)
        (format nil "~a ~a(~a)~a" typ nym (vars-c vars)
                (if body (block-c body) "")))
  (cuda/global (&rest args)
    (format nil "__global__ ~a" (apply #'func-c args)))
  (funcarg (nym varforms)
           (cofy nym)
           (cofsy varforms)
           (format nil "(*~a)(~{~a~^,~})" nym varforms))
  (return (x)
          (cofy x)
          (format nil "return ~a" x))
  (typedef (x y)
           (cofy x)
           (cofy y)
           (format nil "typedef ~a ~a;~%" x y))
  (enum (nym mems)
        (cofy nym)
        (cofsy mems)
        (format nil "enum ~a{~{~a~}};~%" nym mems))
  (h-file (nym)
          (cofy nym)
          (format nil "~a.h" nym))
  (include (filename &key local)
           (cofy filename)
           (format nil "#include ~a~a~a~%" (if local #\" #\<) filename (if local #\" #\>)))
  (import (filename)
    (setf filename (if (stringp filename) filename (format nil "~a.cl" (cof filename))))
    (progn (c-whole-file filename)) (format nil "/* ~a LOADED */" filename))
  (macro (nym &rest xs)
         (cofy nym)
         (format nil "~a(~{~a~^,~})" nym (mapcar #'cof (f/list xs))))
  (define (a b)
          (cofy a)
          (cofy b)
          (format nil "#define ~a ~a~%" a b))
  (paren (x)
         (cofy x)
         (format nil "(~a)" x))
  (comment  (&rest xs)
           (let* ((small (eq (car xs) 's))
                  (s (format nil "/* ~{~a~} */~%" (mapcar #'cof (if small (cdr xs) xs))))
                  (v (if small "" (format nil "/**~{~a~}**/~%" (loop for i from 1 to (- (length s) 7) collect #\*)))))
           (format nil "~%~a~a~a~%" v s v)))
  (header (nym &key local)
          (include-c (h-file-c nym) :local local))
  (headers (&rest xs)
        (format nil "~{~a~}" (mapcar #'(lambda (x) (apply #'header-c (f/list x))) xs)))
  (lisp (x)
        (progn (eval x) ""))
  (lispmacro (f llist &rest body)
             (progn (eval `(cdefun ,f ,llist ,@body)) ""))
  (template (f vars template)
            (progn
              (eval `(cdefun ,f (&rest args)
                             (cof (apply (replacify-lambda ,vars ,template) (mapcar #'cof args)))))
              ""))
  (templates (f vars template)
            (progn
              (eval `(cdefun ,f (&rest argss)
                (block-c (mapcar #'cof (mapcar #'(lambda (args)
                                    (apply
                                      (replacify-lambda ,vars ,template)
                                      (mapcar #'cof (f/list args))))
                                argss)))))))
  (cuda/dim3 (typ x y)
    (cofy typ)
    (cofy x)
    (cofy y)
    (format nil "dim3 ~a(~a,~a)" typ x y))
  (cuda/dim/block (x y)
    (cuda/dim3-c 'dim/block x y))
  (cuda/dim/grid (x y)
    (cuda/dim3-c 'dim/grid x y)))

;; SYNONYMS
(csyn 'cuda/malloc "cudaMalloc")
(csyn 'cuda/memcpy "cudaMemcpy")
(csyn 'cuda/free "cudaFree")
(csyn 'cuda/host->dev "cudaMemcpyHostToDevice")
(csyn 'cuda/dev->host "cudaMemcpyDeviceToHost")
(csyn 'block/idx "blockIdx")
(csyn 'thread/idx "threadIdx")
(csyn 'dim/block "dimBlock")
(csyn 'dim/grid "dimGrid")
(csyn 'null "NULL")
(csyn 'arg/c "argc")
(csyn 'arg/count "argc")
(csyn 'arg/v "argv")
(csyn 'arg/values "argv")
(csyn 'integer "int")
(csyn 'integer+ "long")
(csyn 'real "float")
(csyn 'real+ "double")
(csyn 'boolean "char")
(csyn 'string "char*")

(defun count-lines-in-file (filename)
  (let ((n 0))
    (with-open-file (stream filename :direction :input :if-does-not-exist nil)
            (if stream
                (loop for line = (read-line stream nil 'done)
                    until (eq line 'done)
                    do (incr n))))
    n))
  
  (defun read-whole-file (filename)
    (format nil "~{~a~^~%~}"
        (with-open-file (stream filename :direction :input :if-does-not-exist nil)
            (if stream
                (loop for line = (read-line stream nil 'done)
                    until (eq line 'done)
                    collect line)))))

(defun c-whole-file (filename)
  (let ((s (read-whole-file filename)) (result t) (n 0))
    (apply #'c (loop while result collect
          (progn
            (multiple-value-setq (result n) (read-from-string s nil))
            (setf s (subseq s n))
            result)))))

(defun cwf (filename)
  (format t "~a" (c-whole-file filename)))

(defun tempfilename (&optional extension)
    (labels ((genfilename () (strsof `(temp ,(random 1.0) ,extension))))
            (let ((filename (genfilename)))
              (loop while (probe-file filename) do
                    (setf filename (genfilename)))
              filename)))

(defun compile-cl-file (filein &key fileout tags libs c-file cc)
  (def fileout "a.out")
  (def tags "")
  (def libs "")
  (def cc "gcc")
    (let ((c-code (c-whole-file filein)) (temp-file (if c-file c-file (tempfilename ".c"))))
      (format t "~a" c-file)
      (if (and *last-compiled* (not (eq *last-compiled* c-file))) (delete-file *last-compiled*))
        (with-open-file (c-file-stream temp-file :direction :output :if-does-not-exist :create)
            (format c-file-stream "~a" c-code))
        (format t "Running: ~a ~a ~a -o ~a ~a~%" cc tags temp-file fileout libs)
        (ext:run-shell-command (format nil "~a ~a ~a -o ~a ~a" cc tags temp-file fileout libs))
        (setf *last-compiled* temp-file)))

(defun compile-and-run-cl-file (filein &key args fileout tags libs c-file cc)
    (def fileout "a.out")
    (compile-cl-file filein
      :fileout fileout
      :tags tags
      :libs libs
      :c-file c-file
      :cc cc)
    (format t "Running: ./~a~{~^ ~a~}~%" fileout args)
    (ext:run-shell-command (format nil "./~a~{~^ ~a~}" fileout args)))


(compile 'write-out)
(compile 'change-file)
(compile 'change-exec)
(compile 'compile-c)
(compile 'strof)
(compile 'f/list)
(compile 'f/list/n)
(compile 'strsof)
(compile 'chs->str)
(compile 'str->chs)
(compile 'replace-char)
(compile 'c-strify)
(compile 'addsyms)
(compile 'macn)
(compile 'cnym)
(compile 'c)
(compile 'cof)
(compile 'read-whole-file)
(compile 'c-whole-file)
(compile 'cwf)
(compile 'tempfilename)
(compile 'compile-cl-file)
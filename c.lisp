					; Copyright Jonathan Baca, 2016


(if (not (boundp '  ***C/LISP-SYSTEM-LOADED***))     
 (progn
    (format t "Welcome to LISP/C. Loading constants...")
    (setf *lbrac* #\[)
    (setf *file-out* nil)
    (setf *exec-out* nil)
    (setf *last-compiled* nil)
    (setf *c-synonyms* (make-hash-table))
    (setf *macrolist* (make-hash-table))
    (setf *templatelist* (make-hash-table))))

 (setf ***C/LISP-SYSTEM-LOADED*** T)

                     
(defun pairify (xs)
  (if (null xs)
      nil
      (cons
        (list (car xs) (cadr xs))
        (pairify (cddr xs)))))

(defmacro macropairs (m &rest xs)
  `(progn
     ,@(mapcar #'(lambda (x) `(,m ,@x)) (pairify xs))))

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

(defmacro decr (x)
  `(setf ,x (1- ,x)))

(defun f/list//n (x &optional (n 1))
    (if (<= n 0) x
        (if (atom x)
            (list (f/list//n x (1- n)))
            (if (null (cdr x))
            (list (f/list//n (car x) (- n 2)))
            (list (f/list//n x (1- n)))))))


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

(defun alphap (x)
  (member (char-upcase x) (str->chs "ABCDEFGHIJKLMNOPQRSTUVWXYZ")))

(defun c-strify (x &optional leavemostlyalone)
  (if (stringp x) x
      (let ((s (strof x)))
        (if leavemostlyalone
            (string-downcase s)
	(if (numeric-string s) s
	    (if (eq (char s 0) #\!) (replace-char #\- #\_ (cof (subseq s 1)))
         (if (eq (char s 0) #\=) (camelcase-c (cof (subseq s 1)))
         (if (eq (char s 0) #\-) (Lcamelcase-c (cof (subseq s 1)))
		(replace-char #\- #\_ (string-downcase s))))))))))

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

(defun parenify (x)
  (format nil "(~a)" x))

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

(defun pc (&rest xs)
  (format t "~a" (apply #'c xs)))

(defun repeatn (x &optional (n 1))
  (format nil "~{~a~}"
          (loop for i from 1 to n collect x)))

(defmacro cwrite (&rest xs)
  `(write-out (format nil "~a;~%" (c ,@xs))))

(defun symtrim (x n)
  (read-from-string (subseq (strof x) n)))

(defun capitalize-c (str)
  (format nil "~a~a"
          (string-upcase (char (strof str) 0))
          (string-downcase (subseq (strof str) 1))))

(defun uncapitalize-c (str)
  (format nil "~a~a"
          (string-downcase (char (strof str) 0))
          (subseq (strof str) 1)))

(defun flatten (xs)
  (if (atom xs) (list xs) (mapcan #'flatten xs)))

(defun divide-at (seq elem)
  (labels ((helper (seq elem curr res)
           (if (null seq) (cons (reverse curr) res)
               (if (eq (car seq) elem)
                   (helper
                     (cdr seq) elem nil (cons (reverse curr) res))
                   (helper
                     (cdr seq) elem (cons (car seq) curr) res)))))
         (reverse (helper seq elem nil nil))))

(defun split-str (str ch)
  (remove-if #'(lambda (x) (eq (length x) 0))
             (mapcar #'chs->str (divide-at (str->chs str) ch))))

(defun lowercase-c (&rest strs)
  (format nil "~{~a~}" (mapcar #'string-downcase (mapcar #'strof strs))))

(defun uppercase-c (&rest strs)
  (format nil "~{~a~}" (mapcar #'string-upcase (mapcar #'strof strs))))

(defun camelcase-c (&rest strs)
  (setf strs
        (flatten (mapcan #'(lambda (x) (split-str x #\-)) (mapcar #'strof strs))))
  (setf strs
        (flatten (mapcan #'(lambda (x) (split-str x #\_)) (mapcar #'strof strs))))
  (format nil "~{~a~}" (mapcar #'capitalize-c strs)))

(defun dashify-c (&rest strs)
  (format nil "~{~a~^-~}" (mapcar #'cof strs)))

(defun lcamelcase-c (&rest strs)
  (setf strs
        (flatten (mapcan #'(lambda (x) (split-str x #\-)) (mapcar #'strof strs))))
  (format nil "~a~{~a~}" (string-downcase (car strs)) (mapcar #'capitalize-c (cdr strs))))

(defmacro with-optional-first-arg (args nym default-value possible-values &body body)
  (let ((other (gensym)))
    `(let ((,nym (if (member (car ,args) ',possible-values)
                     (car ,args)
                     ',other)))
       (if (eq ,nym ',other)
          (setf ,nym ,default-value)
          (setf ,args (cdr ,args)))
       ,@body)))
  
(defun gensym-n (&optional (n 1))
  (loop for i from 1 to n collect (gensym)))

(defun bar (&rest xs)
  (with-optional-first-arg xs atmos 'cloudy (cloudy sunny rainy)
      (with-optional-first-arg xs deg 0 (0 1 2 3 4 5)
      (list atmos deg xs))))
(defmacro fib (n)
  (if (< n 2) 1 `(+ (fib ,(1- n)) (fib ,(- n 2)))))

(defun macnx (macro-form &optional (n 1))
  (if (zerop n)
      macro-form
      (if (listp macro-form)
          (if (atom (car macro-form))
              (if (equal (macroexpand-1 macro-form) macro-form)
                  (mapcar #'(lambda (x) (macnx x n)) macro-form)
                  (macnx (macroexpand-1 macro-form) (1- n)))
              (mapcar #'(lambda (x) (macnx x n)) macro-form))
          macro-form)))
                     


(defun cof (x)
  (if (null x)
      ""
      (if (atom x)
          (if (inhash x *c-synonyms*)
              (cof (gethash x *c-synonyms*))
              (c-strify x))
          (if (atom (car x))
              (if (and
                    (> (length (strof (car x))) 1)
                    (not (fboundp (cnym (car x)))))
                    (case (char (strof (car x)) 0)
                        (#\@ (apply #'call-c (cof (symtrim (car x) 1)) (cdr x)))
                        (#\[ (apply #'nth-c (cof (symtrim (car x) 2)) (cdr x)))
                        (#\] (apply #'arr-c (cof (symtrim (car x) 1)) (cdr x)))
                        (#\& (apply #'addr-c (cof (symtrim (car x) 1)) (cdr x)))
                        (#\^ (apply #'cast-c (cof (symtrim (car x) 1)) (cdr x)))
                        (#\* (apply #'ptr-c (cof (symtrim (car x) 1)) (cdr x)))
                        (#\. (apply #'mem-c (cof (symtrim (car x) 1)) (cdr x)))
                        (#\> (apply #'slot-c (cof (symtrim (car x) 1)) (cdr x)))
                        (#\= (apply #'camelcase-c (strof (symtrim (car x) 1)) (mapcar #'strof (cdr x))))
                        (#\% (apply #'lcamelcase-c (strof (symtrim (car x) 1)) (mapcar #'strof (cdr x))))
                        (#\- (apply #'lcamelcase-c (strof (symtrim (car x) 1)) (mapcar #'strof (cdr x))))
                        (otherwise (apply (cnym (car x)) (cdr x))))
                  (apply (cnym (car x)) (cdr x)))
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

;; ## NOW DEFINE THE C LANGUAGE

(binops (=   :l t :nyms (= set let <- ":="))
        (!=  :l t :nyms (!= neq diff different))
        (==  :r t :nyms (== eq same))
        (<   :r t :nyms (< lt))
        (>   :r t :nyms (> gt))
        (<=  :r t :nyms (<= leq le))
        (>=  :r t :nyms (>= geq ge))
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
        (" << " :l t :nparen t :nym <<+) ;; for C++
        (" >> " :l t :nparen t :nym >>+) ;; for C++
        (<<= :l t :nyms (<<= l-shift-eq shift-left-eq shl-eq l-shift= shift-left= shl=))
        (>>  :r t :nyms (>> r-shift shift-right shr))
        (>>= :l t :nyms (>>= r-shift-eq shift-right-eq shr-eq >>= r-shift= shift-right= shr=))
        )

(preposts (++ :post nil :nyms (++  inc +inc incr pre++ +1 ++n))
          (++ :post t   :nyms (+++ pinc inc+ pincr post++ 1+ n++))
          (-- :post nil :nyms (--  dec -dec decr pre-- -1 --n))
          (-- :post t   :nyms (--- pdec dec- pdecr post-- 1- n--))
          (-  :post nil :nyms (neg))
          (!  :post nil :nyms (! not un a flip))
          (~  :post nil :nyms (~ bit-not bit-un bit-a bit-flip))
          (*  :post t   :nyms (ptrtyp arg*) :nparen t))

(cfuns
 (arr-decl (&rest xs)
	   (format nil "{~{~a~^~(, ~)~}}" (mapcar #'cof xs)))
 (sym/add (&rest xs)
	  (cofsy xs)
	  (strsof xs))
 (slot (a &rest bs)
        (cofy a)
        (cofsy bs)
        (format nil "(~a)~a~{~a~^~(->~)~}" a (if bs "->" "") bs))
  (mem (a &rest bs)
        (cofy a)
        (cofsy bs)
        (format nil "(~a)~a~{~a~^.~}" a (if bs "." "") bs))
 (typ* (x &optional (n 1))
       (cofy x)
       (format nil "~a~{~a~}" x (loop for i from 1 to n collect #\*)))
 (const (&rest xs)
	(format nil "const ~a" (apply #'var-c
    (if (= 1 (length xs)) (list (car xs) nil) xs))))
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
 (if (test &optional ifyes ifno)
     (cofy test)
     (cofy ifyes)
     (format nil "if(~a) {~%   ~a;~%}~a"  test ifyes (if ifno (format nil "else{~%   ~a;~%}"(cof ifno)) "")))
 (cond (&rest pairs)
       (format nil "if(~a) {~{~%  ~a;~}~%}~{~a~}" (cof (caar pairs)) (mapcar #'cof (cdar pairs))
         (mapcar #'(lambda (pair) (format nil "else if(~a){~{~%   ~a;~}~%}"
            (cof (car pair)) (mapcar #'cof (cdr pair)))) (cdr pairs))))
 (ifs (&rest pairs)
       (format nil "if(~a) {~{~%  ~a;~}~%}~{~a~}" (cof (caar pairs)) (mapcar #'cof (cdar pairs))
         (mapcar #'(lambda (pair) (format nil "if(~a){~{~%   ~a;~}~%}"
            (cof (car pair)) (mapcar #'cof (cdr pair)))) (cdr pairs))))
 (main (&rest body)
       (format nil "int main(int argc,char **argv)~a" (block-c body)))
 (for (a b c &rest lines)
      (cofy a) (cofy b) (cofy c)
      (format nil "for(~a;~a;~a)~a" a b c (block-c lines)))
 (while (test &rest lines)
   (cofy test)
   (format nil "while(~a) ~a" test (block-c lines)))
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
 (addr (x &optional (n 1))
       (cofy x)
       (format nil "~a(~a)" (repeatn #\& n) x))
 (ptr (x &optional (n 1))
      (format nil "~{~a~}(~a)" (loop for i from 1 to n collect #\*) (cof x)))
 (pt (x &optional (n 1))
     (format nil "~{~a~}~a" (loop for i from 1 to n collect #\*) (cof x)))
 (nth (x &optional (n 0) &rest ns)
      (format nil "(~a)[~a]~a" (cof x) (cof n)
	      (if ns
		  (format nil "~{[~a]~}" (mapcar #'cof ns)) "")))
 (arr (x &optional n &rest ns)
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
      (format nil "\"~{~a~^ ~}\"" x))
 (char (x)
      (cofy x)
      (format nil "'~a'" x))
 (cast (nym &optional (typ 'int) &rest typs)
       (if typs 
	   (apply #'cast-c (cast-c nym typ) typs)
	   (format nil "((~a)(~a))" (cof typ) (cof nym))))
 (var (x &optional type init &rest modifiers)
      (cofy x)
      (cofy type)
      (format nil "~a ~a~{~a~^,~}~a"
              (if modifiers
                  (format nil "~{~a~^ ~}" (mapcar #'cof modifiers))
                  "")
              (if type (format nil "~a " type) "")
              (f/list x) (if init (format nil "=~a" (cof init)) "")))
 (vars (x &optional (inter #\,) (newline t))
       (setf x (mapcar #'(lambda (y) (apply #'var-c (f/list y))) (f/list/n x 1)))
       (format nil (format nil "~~{~~a~~^~(~a~a~)~~}" inter (if newline #\Newline "")) x))
 (varlist (args)
          (vars-c args #\;))
 (struct (nym &optional vars)
   (cofy nym)
   (csyn '***curr-class*** nym)
   (if vars
       (format nil "struct ~a{~%  ~a;~%}" nym (vars-c vars #\;))
       (format nil "struct ~a" nym)))
 (union (nym &optional vars)
   (cofy nym)
   (if vars
       (format nil "union ~a{~%  ~a;~%}" nym (vars-c vars #\;))
       (format nil "union ~a" nym)))
 (block (lines &optional (bracket t))
      (let ((preq ""))
        (if (eq 'const (car lines))
            (progn
              (setf preq " const ")
              (setf lines (cdr lines))))
   (format nil "~a~a~%~{   ~a~^~(;~%~)~};~%~a" preq (if bracket #\{ "")  (mapcar #'cof (f/list lines)) (if bracket #\} "") )))
 (func (nym &optional typ vars &rest body)
       (cofy nym)
       (cofy typ)
       (format nil "~a ~a(~a)~a" typ nym (vars-c vars #\, nil)
	       (if body (block-c body) "")))
 (inline (arg)
         (format nil "inline ~a" (cof arg)))
 (cuda/global (&rest args)
        (format nil "__global__ ~a" (apply #'func-c args)))
 (cuda/device (&rest args)
        (format nil "__device__ ~a" (apply #'func-c args)))
 (funcarg (nym typ &optional varforms)
	  (cofy nym)
    (cofy typ)
	  (cofsy varforms)
	  (format nil "~a(*~a)(~{~a~^,~})" typ nym varforms))
 (return (&optional x)
	 (cofy x)
	 (format nil "return ~a" x))
 (typedef (x &optional y)
	  (cofy x)
	  (format nil "typedef ~a ~a;~%" x (if y (cof y) "")))
 (enum (nym &rest mems)
       (cofy nym)
       (cofsy mems)
       (format nil "enum ~a{~{~a~^~(, ~)~}};~%" nym mems))
 (h-file (nym)
	 (cofy nym)
	 (format nil "~a.h" nym))
 (str/add (&rest xs)
          (format nil "~{~a~}" (cof xs)))
 (include (filename &key local)
	  (cofy filename)
	  (format nil "#include ~a~a~a~%" (if local #\" #\<) filename (if local #\" #\>)))
 (import (filename)
	 (setf filename (if (stringp filename) filename (format nil "~a.cl" (cof filename))))
	 (progn (c-whole-file filename)) (format nil "/* ~a LOADED */" filename))
 (macro (nym &rest xs)
	(cofy nym)
	(format nil "~a(~{~a~^,~})" nym (mapcar #'cof (f/list xs))))
 (unsigned (x)
           (cofy x) 
           (format nil "unsigned ~a" x))
  (define (a b)
     (cofy a)
   (cofy b)
   (format nil "#define ~a ~a~%" a b))
 (ifdef (expr)
        (cofy expr)
      (format nil "#ifdef ~~%" expr))
 (ifndef (expr)
        (cofy expr)
      (format nil "#ifndef ~~%" expr))
 (if# (expr)
        (cofy expr)
      (format nil "#if ~a~%" expr))
 (else# ()
       "#else~%")
 (endif ()
        "#endif~%")
 (pragma (&rest xs)
         (cofsy xs)
         (format nil "#pragma ~{~a~^ ~}" xs))
 (paren (x)
	(cofy x)
	(format nil "(~a)" x))
 (comment  (&rest xs)
           (let* ((small (eq (car xs) 's))
                  (s (format nil "/* ~{~a~^ ~} */~%" (mapcar #'cof (if small (cdr xs) xs))))
                  (v (if small "" (format nil "/**~a**/~%" (repeatn #\* (- (length s) 7))))))
	     (format nil "~%~a~a~a~%" v s v)))
 (header (nym &key local)
	 (include-c (h-file-c nym) :local local))
 (headers (&rest xs)
	  (format nil "~{~a~}" (mapcar #'(lambda (x) (apply #'header-c (f/list x))) xs)))
 (cpp (&rest xs)
      (cofsy xs)
      (format nil "#~{~a~^ ~}" xs))
  (lisp (x)
       (let ((s (eval x)))
           (if (stringp s) s "")))
 (lispmacro (f llist &rest body)
            (if (and
                  (fboundp (cnym f))
                  (not (inhash f *macrolist*)))
      (format nil "/**ERROR: \"~a\" ALREADY DEFINED.**/" f)
      (progn
        (eval `(cdefun ,f ,llist ,@body))
        (sethash f t *macrolist*)
        (format nil "/**DEFINED: \"~a\" (lispmacro)**/" f))))
            
 (lambda (llist template &rest args)
  (cof (eval `(apply (replacify-lambda ,llist ,template) ',args))))
 (template (f vars template)
	     (progn (eval `(cdefun ,f (&rest args)
			    (cof
         (apply (replacify-lambda ,vars ,template)
                (mapcar #'cof args)))))
         (sethash f t *templatelist*)
         (format nil "/**DEFINED: \"~a\" (template)**/" f) ))
 (templates (f vars template)
            (progn
              (eval `(cdefun ,f (&rest argss)
			     (apply #'progn-c (mapcar #'cof (mapcar #'(lambda (args)
								(apply
								 (replacify-lambda ,vars ,template)
								 (mapcar #'cof (f/list args))))
							    argss))))) ""))
 (cuda/dim3 (typ x y)
	    (cofy typ)
	    (cofy x)
	    (cofy y)
    (format nil "dim3 ~a(~a,~a)" typ x y))
 (cuda/dim/block (x y)
		 (cuda/dim3-c 'dim/block x y))
 (cuda/dim/grid (x y)
		(cuda/dim3-c 'dim/grid x y))
 (cuda/shared (&rest xs)
    (format nil "__shared__ ~a" (apply #'var-c xs)))
 (funcall (func &rest args)
    (apply (cnym func) args))
 (apply (func &rest args)
    (setf args (append (butlast args) (car (last args))))
    (apply (cnym func) args))

 ;(args nym default-value possible-values &body body)
  (mapcar (&rest argss)
   (with-optional-first-arg argss brackets? nil (t nil)
      (let ((func (car argss)))
        (setf argss (cdr argss))
        (block-c (apply #'mapcar (cnym func) argss) brackets?))))
 (mapargs (&rest argss)
    (with-optional-first-arg argss brackets? nil (t nil)
      (let ((func (car argss)))
        (setf argss (cdr argss))
        (block-c
          (mapcar
            #'(lambda (args) (apply-c func args))
                 argss) brackets?))))
 (car (&rest args)
  (car args))
 (cdr (&rest args)
  (cdr args))
 (cadr (&rest args)
  (cadr args))
 (cdar (&rest args)
  (cdar args))
 (cddr (&rest args)
  (cddr args))
 (caar (&rest args)
  (caar args))
 (binop (opr &rest xs)
    (cofsy xs)
    (format nil
            (format nil "(~~{(~~a)~~^~~(~a~~)~~})" opr) xs)) 
 (funcall-if (test func &rest args)
      (if test
          (apply #'funcall-c func args)
          (strsof (mapcar #'cof args))))
 (apply-if (test func args)
      (if test
          (apply #'funcall-c func args)
          (strsof (mapcar #'cof args))))
 (test-eq (a b)
          (eq a b))
 (test-not (a)
           (not a))
 (test-and (&rest xs)
          (eval `(and ,@xs)))
 (test-or (&rest xs)
          (eval `(or ,@xs)))
 (code-list (&rest xs)
            (mapcar #'cof xs))
 (list (&rest xs) xs)
)
;; C++ Stuff
(cfuns
  (headers++ (&rest xs)
      (format nil "~{#include<~a>~%~}" (mapcar #'cof xs)))
  (tridot (x)
          (cofy x)
          (format nil "~a..." x))
  (struct++ (nym &rest xs)
      (cofy nym)
    (csyn '***curr-class*** nym)
      (format nil "struct ~a~a" nym (if xs (block-c xs) "")))
  (lambda++ (&optional capture-list params attribs ret &rest body)
            (setf capture-list (mapcar #'(lambda (x) (c-strify x t)) (f/list capture-list)))
            (setf attribs (mapcar #'(lambda (x) (c-strify x t)) (f/list attribs)))
            (cofsy attribs)
      (format nil "[~{~a~^,~}]~a~{~^ ~a~}~a~a"
              capture-list
              (if (or params attribs ret) (parenify (vars-c params #\, nil)) "")
              attribs
              (if ret
                  (format nil " -> ~a " (cof ret)) "")
              (block-c body)))
              
  (namespace (&rest terms)
      (cofsy terms)
      (format nil "~{~a~^~(::~)~}" terms))
  (namespacedecl (nym &rest terms)
     (cofy nym)
      (format nil "namespace ~a~a" nym (block-c terms)))
  (typ& (nym &optional (n 1))
      (cofy nym)
      (format nil "~a~a" nym (repeatn #\& n)))
  (ptr& (nym &optional (n 1))
        (cofy nym)
        (format nil "~a~a" (repeatn #\& n) nym))
  (class (nym &rest terms)
    (cofy nym)
    (csyn '***curr-class*** nym)
    (format nil "class ~a~a" nym (if terms (block-c terms) "")))
  (protected (&rest terms)
    (cofsy terms)
    (format nil "protected:~%~a" (block-c terms nil)))
  (private (&rest terms)
    (cofsy terms)
    (format nil "private:~%~a" (block-c terms nil)))
  (public (&rest terms)
    (cofsy terms)
    (format nil "public:~%~a" (block-c terms nil)))
  (construct (&optional args init-pairs &rest code)
    (format nil "~a(~a)~a~a"
            (cof '***curr-class***)
            (vars-c args)
            (if init-pairs
                (format nil " : ~{~a~^~(, ~)~}"
                    (mapcar #'(lambda (xs)
                        (format nil "~a(~a)"
                                (cof (car xs))
                                (if (cadr xs)
                                    (cof (cadr xs))
                                    (cof (car xs)))))
                      init-pairs))
                "")
            (if code (block-c code) "" )))
  (destroy (&optional args &rest  code)
    (format nil "~~~a(~a)~a"
            (cof '***curr-class***)
            (vars-c args)
            (if code (block-c code) ""))) 
  (operator (oper &optional typ args &rest code)
    (let ((opr "operator"))
            (cofy typ)
            (if (listp oper)
                (if (member (car oper) '(@ ns namespace n/c))
                    (progn
                      (setf opr
                            (apply
                              #'namespace-c
                              (append (butlast (cdr oper)) (list opr))))
                      (setf oper (car (last oper))))))
            (if (null oper) (setf oper "()"))
            (setf oper (string-downcase (strof oper))) 
      (format nil "~a ~a~a~a(~a)~a"
              typ opr (if (alphap (char (strof oper) 0)) " " "") oper (vars-c args) (if code (block-c code) ""))))
  (friend (code)
    (cofy code)
    (format nil "friend ~a" code))
  (decltemp (&optional var typ &rest code)
      (if (listp var)
            (setf var (mapcar #'f/list var))
          (progn
            (setf var (f/list/n (list (list var typ))))
            (setf typ (car code))
            (setf code (cdr code))))
      (cofy typ)
      (setf var (format nil "~{~a~^,~}"
                          (mapcar #'(lambda (pair) (format nil "~{~a~^ ~}"  (reverse (mapcar #'cof pair)))) var)))
      (format nil "template ~a~{~^ ~a~}" (if (or typ var)
                      (format nil "~a<~a>" typ var) "<>")
              (if code (mapcar #'cof code) '(""))))
  (temp (&optional var &rest typs)
        (cofy var) (cofsy typs)
        (format nil "~a<~{~a~^,~}>" var typs))
  (using (namespace)
         (format nil "using namespace ~a" (cof namespace)))
  (usevar (&rest args)
          (format nil "~a" (apply #'var-c (car args) 'using (cdr args))))
  (comment++ (&rest comments)
             (cofsy comments)
             (format nil "//~{~a~^ ~}" comments))
  (new (&rest xs)
       (cofsy xs)
       (format nil "new ~{~a~}" xs))
  (try/catch (catch &rest body)
      (cofy catch)
        (format nil "try~acatch(~a)" (block-c body) catch))
  (strlit (&rest xs)
          (format nil "~a" (apply #'str-c xs)))
  (explicit (&rest xs)
      (cofsy xs)
      (format nil "explicit ~{~a~}"))
)




(macropairs cfunc-syn
func f{}
namespace n/s
namespace ns
namespace @
slot ->
mem .>
typ* t*
typ& t&
ptr p*
ptr& p&
ptr& var&
var v
class c.
class d/c
operator op
operator opr
construct   cx
destroy dx
return r
headers hh
headers++ h+
header h
typedef t/d
nth n.
nth no.
nth nn
arr ar
arr-decl {}s
main m
while w
do-while d/w
for f
arr a.
char ch
str s.
varlist v/l
switch sx
call c
struct s{}
struct sx
struct++ s{}+
struct++ s{+}
struct++ sx+
block b
define d#
pragma p#
public pu.
private pr.
protected px.
friend fr.
template tmplt
template !!
templates !!!
template t.
templates t..
lispmacro l/m
lispmacro !!l
camelcase camel
lcamelcase lcamel
capitalize cap
uncapitalize !cap
lowercase lcase
uppercase ucase
dashify -ify
comment cmt
comment z
comment /*
comment++ cmt+
comment++ cmt++
comment++ z+
comment++ //
temp <>
decltemp <t>
decltemp t<>
<<+ <stream
<<+ <<stream
<<+ <stream<
<<+ stream<
<<+ stream<<
<<+ <<<
>>+ stream>
>>+ stream>>
>>+ >stream
>>+ >>stream
>>+ >>>
addr memloc
addr loc
try/catch t/c
using u.
usevar uv
usevar use
namespacedecl ns/d
namespacedecl n/s/d
namespacedecl ns{}
namespacedecl n/s{}
tridot t---
lambda++ l+)


;; SYNONYMS

;;; CUDA STUFF
(macropairs csyn
'cuda/malloc    "cudaMalloc"
'cuda/memcpy    "cudaMemcpy"
'cuda/free      "cudaFree"
'cuda/host->dev "cudaMemcpyHostToDevice"
'cuda/dev->host "cudaMemcpyDeviceToHost"
'cuda/dev/count "cudaDeviceCount"
'cuda/dev/set   "cudaSetDevice"
'cuda/dev/get   "cudaGetDevice"
'cuda/dev/props "cudaDeviceProperties"
'cuda/sync      "__syncthreads"
'block/idx      "blockIdx"
'block/idx/x    "blockIdx.x"
'block/idx/y    "blockIdx.y"
'block/idx/z    "blockIdx.z"
'thread/idx     "threadIdx"
'thread/idx/x   "threadIdx.x"
'thread/idx/y   "threadIdx.y"
'thread/idx/z   "threadIdx.z"
'block/dim      "blockDim"
'block/dim/x    "blockDim.x"
'block/dim/y    "blockDim.y"
'block/dim/z    "blockDim.z"
'grid/dim       "gridDim"
'grid/dim/x     "gridDim.x"
'grid/dim/y     "gridDim.y"
'grid/dim/z     "gridDim.z"
'dim/block      "dimBlock"
'dim/grid       "dimGrid"

;; MPI STUFF
'mpi/success            "MPI_SUCCESS"
'mpi/err/buffer         "MPI_ERR_BUFFER"
'mpi/err/count          "MPI_ERR_COUNT"
'mpi/err/type           "MPI_ERR_TYPE"
'mpi/err/tag            "MPI_ERR_TAG"
'mpi/err/comm           "MPI_ERR_COMM"
'mpi/err/rank           "MPI_ERR_RANK"
'mpi/err/request        "MPI_ERR_REQUEST"
'mpi/err/root           "MPI_ERR_ROOT"
'mpi/err/group          "MPI_ERR_GROUP"
'mpi/err/op             "MPI_ERR_OP"
'mpi/err/topology       "MPI_ERR_TOPOLOGY"
'mpi/err/dims           "MPI_ERR_DIMS"
'mpi/err/arg            "MPI_ERR_ARG"
'mpi/err/unknown        "MPI_ERR_UNKNOWN"
'mpi/err/truncate       "MPI_ERR_TRUNCATE"
'mpi/err/other          "MPI_ERR_OTHER"
'mpi/err/intern         "MPI_ERR_INTERN"
'mpi/pending            "MPI_PENDING"
'mpi/err/in/status      "MPI_ERR_IN_STATUS"
'mpi/err/lastcode       "MPI_ERR_LASTCODE"
'mpi/bottom             "MPI_BOTTOM"
'mpi/proc/null          "MPI_PROC_NULL"
'mpi/any/source         "MPI_ANY_SOURCE"
'mpi/any/tag            "MPI_ANY_TAG"
'mpi/undefined          "MPI_UNDEFINED"
'mpi/bsend/overhead     "MPI_BSEND_OVERHEAD"
'mpi/keyval/invalid     "MPI_KEYVAL_INVALID"
'mpi/errors/are/fatal   "MPI_ERRORS_ARE_FATAL"
'mpi/errors/return      "MPI_ERRORS_RETURN"
'mpi/max/processor/name "MPI_MAX_PROCESSOR_NAME"
'mpi/max/error/string   "MPI_MAX_ERROR_STRING"
'mpi/char               "MPI_CHAR"
'mpi/short              "MPI_SHORT"
'mpi/int                "MPI_INT"
'mpi/long               "MPI_LONG"
'mpi/unsigned/char      "MPI_UNSIGNED_CHAR"
'mpi/unsigned/short     "MPI_UNSIGNED_SHORT"
'mpi/unsigned           "MPI_UNSIGNED"
'mpi/unsigned/long      "MPI_UNSIGNED_LONG"
'mpi/float              "MPI_FLOAT"
'mpi/double             "MPI_DOUBLE"
'mpi/long/double        "MPI_LONG_DOUBLE"
'mpi/byte               "MPI_BYTE"
'mpi/packed             "MPI_PACKED"
'mpi/float/int          "MPI_FLOAT_INT"
'mpi/double/int         "MPI_DOUBLE_INT"
'mpi/long/int           "MPI_LONG_INT"
'mpi/2int               "MPI_2INT"
'mpi/short/int          "MPI_SHORT_INT"
'mpi/long/double/int    "MPI_LONG_DOUBLE_INT"
'mpi/long/long/int      "MPI_LONG_LONG_INT"
'mpi/ub                 "MPI_UB"
'mpi/lb                 "MPI_LB"
'mpi/comm/world         "MPI_COMM_WORLD"
'mpi/comm/self          "MPI_COMM_SELF"
'mpi/ident              "MPI_IDENT"
'mpi/congruent          "MPI_CONGRUENT"
'mpi/similar            "MPI_SIMILAR"
'mpi/unequal            "MPI_UNEQUAL"
'mpi/tag/ub             "MPI_TAG_UB"
'mpi/io                 "MPI_IO"
'mpi/host               "MPI_HOST"
'mpi/wtime/is/global    "MPI_WTIME_IS_GLOBAL"
'mpi/max                "MPI_MAX"
'mpi/min                "MPI_MIN"
'mpi/sum                "MPI_SUM"
'mpi/prod               "MPI_PROD"
'mpi/maxloc             "MPI_MAXLOC"
'mpi/minloc             "MPI_MINLOC"
'mpi/band               "MPI_BAND"
'mpi/bor                "MPI_BOR"
'mpi/bxor               "MPI_BXOR"
'mpi/land               "MPI_LAND"
'mpi/lor                "MPI_LOR"
'mpi/lxor               "MPI_LXOR"
'mpi/group/null         "MPI_GROUP_NULL"
'mpi/comm/null          "MPI_COMM_NULL"
'mpi/datatype/null      "MPI_DATATYPE_NULL"
'mpi/request/null       "MPI_REQUEST_NULL"
'mpi/op/null            "MPI_OP_NULL"
'mpi/errhandler/null    "MPI_ERRHANDLER_NULL"
'mpi/group/empty        "MPI_GROUP_EMPTY"
'mpi/graph              "MPI_GRAPH"
'mpi/cart               "MPI_CART"
'mpi/aint               "MPI_Aint"
'mpi/status             "MPI_Status"
'mpi/group              "MPI_Group"
'mpi/comm               "MPI_Comm"
'mpi/datatype           "MPI_Datatype"
'mpi/request            "MPI_Request"
'mpi/op                 "MPI_Op"
'mpi/status/ignore      "MPI_STATUS_IGNORE"
'mpi/statuses/ignore    "MPI_STATUSES_IGNORE"
'mpi/copy/function      "MPI_Copy_function"
'mpi/delete/function    "MPI_Delete_function"
'mpi/handler/function   "MPI_Handler_function"
'mpi/user/function      "MPI_User_function"
'mpi/init               "MPI_Init"
'mpi/send               "MPI_Send"
'mpi/recv               "MPI_Recv"
'mpi/bcast              "MPI_Bcast"
'mpi/comm/size          "MPI_Comm_size"
'mpi/comm/rank          "MPI_Comm_rank"
'mpi/abort              "MPI_Abort"
'mpi/get/processor/name "MPI_Get_processor_name"
'mpi/get/version        "MPI_Get_version"
'mpi/initialized        "MPI_Initialized"
'mpi/wtime              "MPI_Wtime"
'mpi/wtick              "MPI_Wtick"
'mpi/finalize           "MPI_Finalize"
'mpi/open/port          "MPI_Open_port"
'mpi/comm/accept        "MPI_Comm_accept"
'mpi/comm/connect       "MPI_Comm_connect"
'mpi/scan               "MPI_Scan"
'mpi/allreduce          "MPI_Allreduce"
'mpi/comm/split         "MPI_Comm_split"
'mpi/isend              "MPI_Isend"
'mpi/irecv              "MPI_Irecv"
'mpi/wait               "MPI_Wait"
'mpi/test               "MPI_Test"
'mpi/init               "MPI_Init"
'mpi/finalize           "MPI_Finalize"
'mpi/comm/rank          "MPI_Comm_rank"
'mpi/comm/size          "MPI_Comm_size"
'mpi/get/count          "MPI_Get_count"
'mpi/type/extent        "MPI_Type_extent"
'mpi/type/struct        "MPI_Type_struct"
'mpi/scatter            "MPI_Scatter"
'mpi/gather             "MPI_Gather"
'mpi/sendrecv           "MPI_Sendrecv"
'mpi/sendrecv/replace   "MPI_Sendrecv_replace"
'mpi/group/rank         "MPI_Group_rank"
'mpi/group/size         "MPI_Group_size"
'mpi/comm/group         "MPI_Comm_group"
'mpi/group/free         "MPI_Group_free"
'mpi/group/incl         "MPI_Group_incl"
'mpi/comm/create        "MPI_Comm_create"
'mpi/wtime              "MPI_Wtime"
'mpi/get/processor/name "MPI_Get_processor_name"


;;; PTHREADS API
`pthread/create              "pthread_create"
`pthread/equal               "pthread_equal"
`pthread/exit                "pthread_exit"
`pthread/join                "pthread_join"
`pthread/self                "pthread_self"
`pthread/mutex/init          "pthread_mutex_init"
`pthread/mutex/destroy       "pthread_mutex_destroy"
`pthread/mutex/lock          "pthread_mutex_lock"
`pthread/mutex/trylock       "pthread_mutex_trylock"
`pthread/mutex/unlock        "pthread_mutex_unlock"
`pthread/cond/init           "pthread_cond_init"
`pthread/cond/destroy        "pthread_cond_destroy"
`pthread/cond/wait           "pthread_cond_wait"
`pthread/cond/timedwait      "pthread_cond_timedwait"
`pthread/cond/signal         "pthread_cond_signal"
`pthread/cond/broadcast      "pthread_cond_broadcast"
`pthread/once                "pthread_once"
`pthread/key/create          "pthread_key_create"
`pthread/key/delete          "pthread_key_delete"
`pthread/setspecific         "pthread_setspecific"
`pthread/getspecific         "pthread_getspecific"
`pthread/cleanup/push        "pthread_cleanup_push"
`pthread/cleanup/pop         "pthread_cleanup_pop"
`pthread/attr/init           "pthread_attr_init"
`pthread/attr/destroy        "pthread_attr_destroy"
`pthread/attr/getstacksize   "pthread_attr_getstacksize"
`pthread/attr/setstacksize   "pthread_attr_setstacksize"
`pthread/attr/getdetachstate "pthread_attr_getdetachstate"
`pthread/attr/setdetachstate "pthread_attr_setdetachstate"
`flockfile                   "flockfile"
`ftrylockfile                "ftrylockfile"
`funlockfile                 "funlockfile"
`getc/unlocked               "getc_unlocked"
`getchar/unlocked            "getchar_unlocked"
`putc/unlocked               "putc_unlocked"
`putc/unlocked               "putc_unlocked"
`pthread/detach              "pthread_detach"
'pthread/threads/max         "PTHREAD_THREADS_MAX"
'pthread/keys/max            "PTHREAD_KEYS_MAX"
'pthread/stack/min           "PTHREAD_STACK_MIN"
'pthread/create/detached     "PTHREAD_CREATE_DETACHED"
'pthread/create/joinable     "PTHREAD_CREATE_JOINABLE"

;;; BASIC STUFF
'null       "NULL"
'arg/c      "argc"
'arg/count  "argc"
'arg/v      "argv"
'arg/values "argv"
'size/t     "size_t"
'integer    "int"
'integer+   "long"
'natural    "unsigned int"
'natural+   "unsigned long"
'real       "float"
'real+      "double"
'boolean    "char"
'stringc    "char*"
'---        "..."
'-#         "#"
'-##        "##"
'-va-args-  "__VA_ARGS__"
'-empty-    " "
'--         " "
'-          "_"
'$          nil)

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

(defun stamp-time ()
  (format nil "~{~a~^/~}"
    (reverse (multiple-value-list (get-decoded-time)))))

(defun c-whole-file (filename)
  (let ((s (read-whole-file filename)) (result t) (n 0))
    (format nil "/*~a*/~%~a"
            (stamp-time)
      (apply #'c (loop while result collect
		    (progn
		      (multiple-value-setq (result n) (read-from-string s nil))
		      (setf s (subseq s n))
		      result))))))

(defun cwf (filename)
  (format t "~a" (c-whole-file filename)))

(defun tempfilename (&optional extension)
  (labels ((genfilename () (strsof `(temp ,(random 1.0) ,extension))))
    (let ((filename (genfilename)))
      (loop while (probe-file filename) do
	   (setf filename (genfilename)))
      filename)))

(defun fileext (nym ext)
  (format nil "~a.~a" (c-strify nym) (c-strify ext)))

(defun c-cl-file (filein &optional fileout)
  (let ((s (c-whole-file filein)) (temp nil))
    (if (null fileout)
        (progn
          (setf temp filein)
          (setf filein (fileext temp 'cl))
          (setf fileout (fileext temp 'c))))
    (if s
  (with-open-file (c-file-stream fileout :direction :output :if-does-not-exist :create)
    (format c-file-stream "~a" s)))))

(defun elapsed-time (sec)
  (let* ((min (floor (/ (floor sec) 60)))
         (hr  (floor (/ min 60)))
         (day (floor (/ hr  24))))
    (setf sec (floor sec))
    (setf sec (mod sec 60))
    (setf min (mod min 60))
    (setf hr (mod hr 24))
    (format nil "~a~a~2,'0d:~2,'0d"
            (if (zerop day) "" (format nil "~a days, " day))
            (if (zerop hr) "" (format nil "~2,'0d:" hr))
            min sec)))

(defun c-cl-file-continuous (filein &optional fileout ie (interval 1))
  (format t "Press ^C to stop.")
    (do ((i 0 (+ i interval))) (nil)
        (progn
          (format t "~&~a" (elapsed-time i))
          (if ie
              (ignore-errors (c-cl-file filein fileout))
              (c-cl-file filein fileout))
          (sleep interval))
        ))
      

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
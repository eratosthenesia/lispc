(header stdio)
(header stdlib)

(typedef (struct linked-list-node) linked-list-node)
(typedef (struct linked-list) linked-list)

(struct linked-list-node
    (
     ((pt prev) linked-list-node)
     ((pt next) linked-list-node)
     ((pt data) void)
     ))

(comment "this is where the linked list is defined")
(struct linked-list
	(
        ((pt first) linked-list-node)
        ((pt last)  linked-list-node)
        length
	))

(func (pt new-linked-list-node) linked-list-node (((pt prev) linked-list-node) ((pt next) linked-list-node) ((pt data) void))
    (var (pt retval) linked-list-node)
    (set retval (@malloc (@sizeof linked-list-node)))
    (set (slot retval prev) prev)
    (set (slot retval next) next)
    (set (slot retval data) data)
    (return retval))

(func (pt new-linked-list) linked-list ()
    (var (pt retval) linked-list)
    (set retval (@malloc (@sizeof linked-list)))
    (set (slot retval first) (slot retval last) (@new-linked-list-node !null !null !null))
    (set (slot retval length) 0)
    (return retval))

(func map-linked-list void (((pt list) linked-list) ((pt (funcarg func ((arg* void) (arg* void)))) void) ((pt etc) void))
      (var (pt cursor) linked-list-node)
      (for (set cursor (slot list first)) cursor (set cursor (slot cursor next))
	   (call (paren (ptr func)) (slot cursor data) etc)
	   ))

(func (pt listify-linked-list 2) void (((pt list) linked-list)))
(func (pt listify-linked-list- 2) void (((pt curr) linked-list-node) n))


(comment "Produces null-terminated list.")


(func (pt listify-linked-list 2) void (((pt list) linked-list))
    (return (@listify-linked-list- (slot list first) 0)))
(func (pt listify-linked-list- 2) void (((pt curr) linked-list-node) n)
      (var (pt ret 2) void)
      (comment s "Allocate ret as appropriate, then set it.")
      (if curr 
	  (progn
            (set ret (@listify-linked-list- (slot curr next) (+ n 1)))
            (set (nth ret n) !NULL))
        (progn
	  (set ret (@malloc (* n (@sizeof (arg* void)))))
	  (set (nth ret n) (slot curr data))))
      (return ret))



(comment "Deletes the list")
(func (pt delete-linked-list) void (((pt list) linked-list) (free-data char)))
(func (pt delete-linked-list-) void (((pt curr) linked-list-node) (free-data char)))

(func (pt delete-linked-list) void (((pt list) linked-list) (free-data char))
      (@delete-linked-list- (slot list first) free-data))
(func (pt delete-linked-list-) void (((pt curr) linked-list-node) (free-data char))
      (if free-data (@free (slot curr data)))
      (@delete-linked-list- (slot curr next) free-data)
      (@free curr))


(define !ARRLEN 12)

(template second (x) (nth x 2))
(comment "main")
(main
 (var (arr x !ARRLEN) float)
 (var i)
 (set (second x) 23.45f)
 (@printf (str "Currently, the second value of x is %f\\n") (second x))
 (for (let i 0) (lt i !ARRLEN) (inc i)
      (set ([]x i) (/ (cast i float) 2))
      (@printf (str "%f %d\\n") ([]x i) i))
 (return 0))

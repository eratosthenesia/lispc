# __LISP__/__c__

## Installing

To install, simply go into the directory that you downloaded everything into, run `clisp`, and type `(load "c.lisp")`. To compile a `cl` file into a `c` file, type `(c-cl-file source.cl dest.c)`. To compile and run a `cl` file, type in `(compile-and-run-cl-file file.cl)`. More documentation on this part to come. <sup><sub>TODO</sub></sup>

## Resources

To learn **C**, I recommend *The C Programming Language* by **Brian W. Kernighan** (ISBN-10 0131103628, ISBN-13 978-0131103627). TO learn **LISP**, I recommend *Practical Common Lisp* by **Peter Seibel**. This can be found either [here](www.gigamonkeys.com/book/) or as a hard copy (ISBN-10 1590592395, ISBN-13 978-1590592397). Also, it is currently required that you use **CLISP** to run the code here.. This will change <sup><sub>**TODO**</sub></sup>.

To learn **CUDA**, I recommend the resources found [here](https://developer.nvidia.com/cuda-education-training), and to learn **MPI**, I recommend the resources found [here](http://mpitutorial.com/tutorials/).

## Introduction
**LISP**/**c** is a powerful macrolanguage for **C**. It basically turns this:

    (header stdio)
    (main
      (@printf (str "Hello, world!"))
      (return 0))

into (after it being cleaned up (more on this later)) this:

    #include <stdio.h>

    int main(int argc,char **argv)
    {
       printf("Hello, world!");
       return 0;
    }

But why?

## Why

Because __LISP__ is expressive and __C__ is fast and I wanted the best of both worlds is the short answer. The longer answer has something to do with macros. But instead of immediately boring you with that, I'll answer what you really want to know:

## Why Should I Care?

First let's discuss if you *can* use it. Not to be elitist (I wish everyone would use this tool), but you **must** know both **C** and **LISP** fairly well to be able to use __LISP__/__c__.

Suppose, however, that you do already know both __LISP__ and __C__ pretty well. You might want to use __LISP__/__c__ because it features access to __LISP__ to write __C__ code obth implicitly and explicity. You might also want to use it if you like writing **CUDA** code, because it has built-in support for **CUDA** as well.

But really, to see why you might like to use __LISP__/__c__, check out a few examples, and feel free to skip around a little.

## An Example

Suppose that you're writing a function that you'd like to write for several different types of vaariable types that use similar notation. You can do this easily with the templates built into **LISP**/**c**:

    int foo_int(int x, int y) {
      return x + y * 2;
    }
    float foo_float(float x, float y) {
      return x + y * 2;
    }
    // etc.

It's true that you can just use a long macro in **C** to get rid of the annoying task, but it's a bit awkward. You can do the same in **LISP**/**c** using the following notation (with `template`):

    (template make-foo (typ)
      (func (add/sym foo- typ) typ ((x typ) (y typ))
        (return (+ x (* y 2)))))
    (make-foo int) (make-foo long) (make-foo float) (make-foo etc)

Or even like this (with `template`__s__):

    (templates make-foo (typ)
      (func (add/sym foo- typ) typ ((x typ) (y typ))
        (return (+ x (* y 2)))))
    (make-foos int long float etc)

And just like that, you have a bunch of functions written. Now to get you sort of grounded, let's go through this element by element.

### Arithmetic

In the amove example, you'll notice that we use prefix arithmetic. This is a feature of **LISP** and not of **C**. The benefit of using prefix arithmetic is that it allows you to express sums of many terms somewhat more succinctly. That is to say, instead of
`2 + 3 + 4 + 5 * 6 + 7 + 8` you can just write `(+ 2 3 4 (* 5 6) 7 8)`.

### Functions

Functions have the general form:

    (func function-name return-type (variables...) body...)

and convert to

    return_type function_name(variables...) {
      body...
    }

If you need a function which returns a pointer to something, you can use:

    (func (pt function-name 2) return-type ...)

Which turns into

    return_type **function_name(...) {...}

Do note that the `2` is required because there are two `*`s, but if there were only one, you could just use `(pt function-name)`. That's the flexibility that makes __LISP__/__c__ nice to work with.

There are two ways that functions can be called. Suppose we want to know the value of `foo(2,3,4)`. We can either use:

    (call foo 2 3 4)

or

### The @ Notation

    (@foo 2 3 4)

This is the same thing as `(call foo 2 3 4)`. This is used to greatly simplify function calls. Use this whenever possible, since nobody wants to wade through a bunch of `call` statements. `call` is mainly useful for `template` statements.

### Thing Names

Variable, type, function, etc. (identifier) names are converted using some simple rules. First of all, the `-`s are turned into `_`s. Secondly, everything is lower case. If you need to make something upper case entirely, you can prefix it with a `!` (so if you need the variable name `CONST_FOO` you can use `!const-foo`, `!cOnST-FoO`, `!const_FOO` or `"CONST_FOO"`). The last one may be used because strings are preserved. The others work because **LISP** is not case-sensitive, so when the variables are rendered, all case is the same. So if you were to use `cOnST-FoO` instead of `!cOnST-FoO`, you'd wind up with `const_foo` instead of `CONST_FOO`.

### Continuing Forward

We need some sort of framework for showing each of the features of __LISP__/__c__. So before I go through every function and explain what it does, I'm going to explain a little of what goes on behind the scenes.

## Engine

The main file for interacting with __LISP__/__c__ right now is just using __CLISP__ (for the time being; it will be ported to more versions soon) and typing in `(load "c.lisp")`.

You can test out the engine by either loading in a file using the `cwf` command and typing `(cwf "example.cl")`. What you'll see is either an error (because of syntax) or the resultant **C** code. If you don't have a file that you can experiment with yet, try typing the following:

     (c '(typedef (struct foo (
             bar
             ((pt baz) float) )) qux))
It will result in the following (or similar):
I
    typedef
    struct foo{
    int bar;
    float *baz;} qux;

This, cleaned up, is:

    typedef struct foo {
        int bar;
        float *baz;
    } qux;

Future versions of **LISP**/**c** will have nicer-looking output **C** code.

The way that **LISP**/**c** works is sort of tricky. It utilizes a lot of string manipulation and interpretation. It doesn't need to be insanely fast, though, because it's just dealing with code; the compiled result will not suffer from any slowness.

## Top-Level Functions

These are the functions that are to be run directly from your **LISP** REPL environment.

### `(cwf` filename`)`
Prints the compiled **C** file from the *filename* containing your **LISP**/**c** code.

### `(compile-cl-file` file-in ...arguments... `)`
This uses **gcc** to compile your file (at *file-in*). It takes a number of keyword arguments (expressed as `:keyword argument`):

| Keyword | Argument |
| --- | --- |
| fileout | executable output |
| tags | tags for compilation |
| libs | libraries for compilation |
| c-file | output **C** file |
| cc | compiler to use |

### `(compile-and-run-cl-file` ... `)`
Uses the same syntax as `compile-cl-file`.

### `(c-cl-file file-in c-file `)`
Compiles **LISP**/**c** code into **C** code from *file-in* to *c-file*.

### Other conventions
When a **LISP**/**c** function such as `while` is called, it's actually calling a lisp function called `while-c`. This may change in the future, but is done presently for convenience. <sup><sub>**TODO**</sub></sup>

## An Example: Multithreading

Suppose you want to do some threading using `pthreads`. You'd start with the headers obviously:
    
    (headers stdio stdlib string pthread)
    
Not all of these are required, but I included all of them to show that you can. It compiles to the following:

    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <pthread.h>

Next, we know that we're going to want to create and then join a bunch of threads using a `for` structure that is almost the same in both cases. Rather than have code duplication on our consciousnesses, we can write a `template` to take care of this for us:

    (TEMPLATE Loop-N (n v maxv body) ... )

We'll finish this in a moment. I changed up the capitalization again just to make it crystal clear that *that is a thing you can do and must be aware of*. Here, `n` will be the number of iterations, `v` will be the variable that we're keeping track with, `maxv` will be a temporary variable that we store `n` in (so that if we need to calculate `n` we're not doing it every time), and `body` is the body of the `for` loop.

FInishing up the function, we yield:

    (TEMPLATE Loop-N (n v maxv body)
        (block (
            (var v    Integer 0) ;; Integer is a synonym for "int"
            (var maxv Integer n)
            (for () (< v maxv) (++ v)
                body))))
                
You'll notice that `Integer` is being used in lieu of "int". Next we're going to use templates instead of macros to convert integers to voids and vice versa:

    (template void<-int (x) (cast x size-t (typ* void))))
    (template int<-void (x) (cast x size-t int))

Here `(typ* void)` expands to `void*`. Basically this converts between `int` and `void*`. The reason why I chose to use a template instead of a **C** macro is because of freedom of notation; I wanted to use the arrows.

Next we write the thread function:

    (func (pt threadfunc) void               ; void *threadfunc
        ( ((pt x) void) )                    ; (void *x) {
        (var i int (int<-void x))            ; int i = (int)(size_t)x;
        (@printf                             ; printf(
            (str "Hello from thread %d\\n")  ; "Hello from thread %d",
            (+ 1 i))                         ; i+1);
        (return null))                       ; return NULL;}

Hopefully this is fairly self-explanatory. Finally, we write the main function:

    (define !nthreads 12) ;; #define NTHREADS 12
    (main
        (var (arr threads !nthreads) pthread-t)
        ;; pthread-t threads[NTHREADS];
        (loop-n !nthreads i maxi    ;; loop for i from 1 to NTHREADS...
            (@pthread-create ;; pthread_create(
                (addr ([]threads i)) ;; &threads[i],
                null                 ;; NULL,
                (addr threadfunc)    ;; &threadfunc,
                (void<-int i)))      ;; (void*)(size_t)i);
        (loop-n !nthreads i maxi    ;; loop for i from 1 to NTHREADS
            (@pthread-join (nth threads i) null))
            ;; pthread_join(threads[i], NULL);
        (return 0)) ;; return 0;}

## CUDA Example

Here is an adapted version of NVIDIA's code for the Julia set:

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
        (var c (struct cu-complex) (cu-complex-decl -0.8 0.156))
        (var a (struct cu-complex) (cu-complex-decl jx jy))
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

This generates the following **C** code (after being cleaned up):
    
    #include "../common/book.h.h"
    #include "../common/cpu_bitmap.h.h"

    #define DIM 1000
    
    struct cu_complex{
    float r;
    float i;
    };
    
    float cu_complex_magnitude(cu_complex x)
    {
       return (((((x).i)*((x).i)))+((((x).r)*((x).r))));
    };
    
    cu_complex cu_complex_mul(cu_complex x,cu_complex y)
    {
       cu_complex z;
       (((z).r)=((((((x).r)*((y).r)))-((((x).i)*((y).i))))));
       (((z).i)=((((((x).i)*((y).r)))+((((x).r)*((y).i))))));
       return z;
    };
    
    cu_complex cu_complex_add(cu_complex x,cu_complex y)
    {
       cu_complex z;
       (((z).r)=((((x).r)+((y).r))));
       (((z).i)=((((x).i)+((y).i))));
       return z;
    };
    
    __device__ 
    int julia(int x,int y)
    {
       const float scale=1.5;
       ;
       float jx=((scale)*(((int)(((((((DIM)/(2)))-(x)))/(((DIM)/(2))))))));
       float jy=((scale)*(((int)(((((((DIM)/(2)))-(x)))/(((DIM)/(2))))))));
       ;
       struct cu_complex c=((struct cu_complex)({-9.8, 0.156}));
       struct cu_complex a=((struct cu_complex)({jx, jy}));
       int i=0;
       
    for(((i)=(0));((i)<(200));++(i))
    {
       ((a)=(cu_complex_add(cu_complex_mul(a,a),c)));
       
    if(((cu_complex_magnitude(a))>(1000))) {
       return 0;
    };
    };
       return 1;
    };
    
    __global__ 
    void kernel(unsigned char *ptr)
    {
       int x=blockIdx.x;
       int y=blockIdx.y;
       int offset=((x)+(((y)*(gridDim.x))));
       int julia_value=julia(x,y);
       (((ptr)[((0)+(((offset)*(4))))])=(((255)*(julia_value))));
       (((ptr)[((1)+(((offset)*(4))))])=(0));
       (((ptr)[((2)+(((offset)*(4))))])=(0));
       (((ptr)[((3)+(((offset)*(4))))])=(255));
    };
    
    ;
    
    int main(int argc,char **argv)
    {
       CPUbitmap bitmap(DIM,DIM);
       unsigned char *dev_bitmap;
       HANDLE_ERROR(cudaMalloc(((void**)(&(dev_bitmap))),(bitmap).image_size()));
       dim3 grid(DIM,DIM);
       kernel<<<grid,1>>>(dev_bitmap);
       HANDLE_ERROR(cudaMemcpy((bitmap).get_ptr(),dev_bitmap
                    (bitmap).image_size(),cudaMemcpyDeviceToHost));
       (bitmap).display_and_exit();
       HANDLE_ERROR(cudaFree(dev_bitmap));
    };
    


## MPI Example

__LISP__/__c__ has support for **MPI** as well. For example, the following program:

    (headers
        (mpi :local t)
        stdio)
    
    (main
        (vars (numtasks rank len rc))
        (var (arr hostname mpi/max/processor/name) char)
        (set rc (@mpi/init (addr argc) (addr argv)))
        (if (neq rc mpi/success)
            (progn
                (@printf
                    (str "Error starting MPI program. Terminating.\\n"))
                (@mpi/abort mpi/comm/world rc)))
        (@mpi/comm/size mpi/comm/world (addr numtasks))
        (@mpi/comm/rank mpi/comm/world (addr rank))
        (@mpi/get/processor/name hostname (addr len))
        (@printf
            (str "Number of tasks= %d My rank = %d Running on %s\\n")
            numtasks rank hostname)
        (@mpi/finalize)
    )

Compiles to the example program:

    #include "mpi.h"
    #include <stdio.h>
    ;
    
    int main(int argc,char** argv)
    {
       
    int numtasks,
    int rank,
    int len,
    int rc;
       char hostname[MPI_MAX_PROCESSOR_NAME];
       ((rc)=(MPI_Init(&(argc),&(argv))));
       
    if(((rc)!=(MPI_SUCCESS))){
       
      printf("Error starting MPI program. Terminating.\n");
      MPI_Abort(MPI_COMM_WORLD,rc);;
    };
       MPI_Comm_size(MPI_COMM_WORLD,&(numtasks));
       MPI_Comm_rank(MPI_COMM_WORLD,&(rank));
       MPI_Get_processor_name(hostname,&(len));
       printf("Number of tasks= %d My rank = %d Running on %s\n",numtasks,rank,hostname);
       MPI_Finalize();
    };

## Synonyms

There are a lot of synonyms present in **LISP**/**c**. For example, you may type `integer` instead of `int` or `integer+` instead of `long int`. A full list of synonyms can be found in the source code for **LISP**/**c**.

## A List of Functions

For your convenience, the full list (so far) of functions defined (*and* documented) in **LISP**/**c** are `?`, `arr`, `arr-decl`, `block`, `call`, `cast`, `char`, `comment`, `cond`, `const`, `cuda/call`, `cuda/device`, `cuda/global`, `cuda/shared`, `define`, `do-while`, `enum`, `for`, `func`, `funcarg`, `h-file`, `header`, `headers`, `if`, `import`, `include`, `lisp`, `lispmacro`, `macro`, `main`, `nth`, `paren`, `pragma`, `progn`, `pt`, `ptr`, `return`, `str`, `struct`, `switch`, `sym/add`, `syn`, `template`, `templates`, `typ*`, `typedef`, `unsyn`, `var`, `varlist`, `vars`, and `while`.

These are functions which exist within **LISP**/**c**:

### `(arr-decl` val<sub>1</sub> ... val<sub>n</sub> `)`
This function declares a literal array of values. It compiles to the **C** code `{`val<sub>1</sub>`,`...`,`val<sub>n</sub>`)`.

### `(sym/add` val<sub>1</sub> ... val<sub>n</sub> `)`
This creates a new identifier that is an aggregate of the individual identifiers, as they have been compiled. This works well in `template` statements.

### `(typ* ` type {n (default = 1)}? `)

This creates a pointer type. For example, `(typ* integer)` compiles to `int*`, and `(typ* char 4)` compiles to `char****`.

### `(var` var {type (default = int)}? {init}? {modifiers}* `)`
Declares a variable. If init is specified, it compiles to a declaration of that variable with that type.

### `(const` ... `)`
Uses the same arguments as `var`, but puts a `const` at the beginning automatically. Equivalent to `(var ... const)`.

### `(syn` term synonym `)`
Looks at both *term* and *synonym* and declares that any instance of *term* by itself will compile to *synonym*.

### `(unsyn` term `)`
Declares that *term*, if it is supposed to compile to any synonym, will no longer do so.

###  `(progn` lines `)`
This just puts a bunch of lines in the slot where one thing should go. Useful in if-else statements.

### `(?` test if-true if-false `)`
This compiles to a `?:` statement. It compiles directly to `(`test`)?:(`if-true`):(`if-false`)`,

### `(if` test if-true if-false `)`
Like the above, but compiles to an if statement.

### `(cond` {`(`condition if-true`)`}* `)`
Works like the `cond` statement in **LISP**, but for **C**. Does this with a series of if-else statements.

### `(main` {statements}* `)`
Creates the main function.

### `(for ` start continue-test step {statements}* `)`
This compiles to a `for` statement in **C**.

### `(while` test {statements}* `)`
Creates a `while` statement.

### `(do-while` test {statements}* `)`
Creates a `do...while` statement, but puts the test at the end where it belongs.

### `(switch` variable {value if-equal {break}?}* `)`
This creates a switch statement. There is no special treatment of the `default` clause. If any of the tuples cntaining the value and the if-equal statement has a third argument (which it does not have to), and that value is anything other than `nil`, it puts a `break;` statement into the compiled **C**.

### `(ptr` x {n (default = 1)}? `)`
This dereferences `x` `n` times. For example `(ptr a 2)` compiles to `**(a)`.

### `(pt` ... `)`
This uses the same syntax as `ptr` does, but it does not put parentheses around the *x* in question.

### `(nth` value index {indices}* `)`
This gets the index<sup>th</sup> reference of value. For example, `(nth a b)` compiles to `(a)[b]`, and `(nth a b c)` compiles to `(a)[b][c]`.

### `(arr` ... `)`
This uses the same syntax as `nth`, but does not put parenthesis around the *value& in question.

### `(call` function-name {arguments}* `)`
This simply calls function-name with arguments.

### `(cuda/call` function-name spec-list {arguments}* `)`
This calls the **CUDA** function with the name *function-name* with the specifications *spec-list*. For exmaple, `(cuda/call foo (16 32) a b c)` compiles to `foo<<<16,32>>>(a,b,c)`.

### `(str` {values}* `)`
This strings together all the values with spaces between them and formats them as a `cstring`. Like `(str a b "cDe")` compiles to `"a b cDe"`.

### `(char` value `)`
Formats *value* as a `char` For example `(char x)` compiles to `'x'`, `(char \\n)` compiles to `'\n'`, and `(char "X")` compiles to `X`.

### `(cast` value type {types}* `)`
This casts *value* as *type*, and if *types* are specified, then if casts them as those too, but in "reverse" order. For example, `(cast x abc)` compiles to (after code is cleaned up) `(abc)x`, and `(cast x abc def)` compiles to `(def)(abc)x`,

### `(vars` specs-ilsts `)`
A bunch of variables, comma-separated, with the arguments to each one supplied by an entry in *specs-lists*.

### `(varlist` ... `)`
Uses the same syntax as `vars`, but puts semicolons between the variable delcarations.

### `(struct` struct-name ({variables}*) `)`
Creates a structure named *struct-name* with variables *variables*. For example:

    (struct foo (
        bar
        (baz qux)
        ((pt xyzzy) foobar)))

Compiles to

    struct foo {
        int bar;
        qux baz;
        foobar *xyzzy;
    };

### `(block` linelist {bracket? (default = t)}? `)`
This creates a **C** block structure. If *bracket* is set to `nil`, then it has no brackets around it. This serves mainly as a way to consolidate elements generated for `template` recipes.

### `(func` name type variables {body}* `)`
This creates a function with name *name, type *type*, variables *variables* (as processed through the `vars` facility), and with code inside *body*. If *body* is not specified, then there is no code inside the function and it is treated as a function prototype. If variables is set to `()` or `nil`, then the variable list will be compiled in **C** as `()`. There is currently no facility for a `void` specification, but that will change shortly. <sub><sup>**TODO**</sup></sub>

### `(cuda/global` ... `)`
Uses the same syntax as `func`, but appends `__global__` to the beginning.

### `(cuda/device` ... `)`
Uses the same syntax as `func`, but appends `__device__` to the beginning.

### `(funcarg` name type variables `)`
This creates a function argument with name *name*, type *type*, and variables *variables*. For example, `funcarg foo bar (int (arg* float)))` compiles to `bar(*foo)(int,float*)`.

### `(return` value `)`
Creates a `return` statement that returns *value*.

### `(typedef` old-type new-type `)`
Creates a simple typedef statement. For example, `(typedef (arg* int) intptr)` compiles to `typedef int* intptr;`

### `(enum` enum-name {specs}* `)`
This creates an enum with the name *enum-name* and specifications *specs*. For example, `(enum a (b c d))` compiles to `enum a{b, c, d}`.

### `(h-file` name `)`
Outputs name.h.

### `(include` name {`local:` local (default = nil)}? `)`
Includes a .c or .h file with the name *name*. If local is spcified, then `"` are used instead of `<>`.

### `(import` filename `)`
Imports a .cl file with name *fllename* (if *filename* is not a string, then `.cl` is appended). This is the **LISP**/**c** version of `#include`. So far, it does not keep track of directories, so all files included, including files included in files included must be in the same directory. <sup><sub>**TODO**</sub></sup>

### `(macro` macro-name {macro-args}* `)`
This creates a simple funcall-type structure, but is meant to be used with `define`. It was defined early on in devlopment and may be phased out.

### `(define` definer definee `)`
Makes a `#define` statement in **C** with *definer* being the dirst argument and *definee* being the second argument.

### `(pragma` {statements}* `)`
Makes a `#pragma` statement in **C** with each statement seperated by a space.

### `(paren` term `)`
Puts parentheses around *term*.

### `(comment` {comments}* `)`
*Comments*, separated by spaces, are put into a comment form like the following:

THe following:

    (comment this is "A Comment")
    (comment s this is "A Comment")

compile to (respectively):

    /*********************/
    /* this is A Comment */
    /*********************/
    
    /* this is A Comment */

The reason why the second comment was shorter was because it began with an `s`.

### `(header` name {`local:` local (default = nil)}? `)`
Same as `include`, but automatically adds a `.h` to the end of *name*.

### `(headers` argument-lists `)`
Each argument in the argument list can be an atom, which will be assumed to be a list in the final phase of processing. For exmaple,

    (headers foo (bar :local t))
    
compiles to the **C** code

    #include <foo.h>
    #include "bar.h"

This is useful if you have a whole slew of things to include. It's also worth noting that something like `(headers arpa/inet)` will compile to `#include <arpa/inet.h>`.

### `(lisp` lisp-code `)`
Runs **LISP** code directly. For very low-level maintenance. ***DO NOT USE UNLESS YOU KNOW WHAT YOU ARE DOING***.

### `(lispmacro` name arglist {body}* `)`
This creates a function in **LISP** directly with a name callable by **LISP**/**c** code as *name*. Again, ***THIS IS ONLY TO BE TOUCHED BY PEOPLE WHO KNOW WHAT THEY'RE DOING. YOU CAN SCREW UP THE WHOLE ENGINE.***

### `(template` name arguments form `)`
Creates a new function with the name *name* with arguments *arguments* and form *form*. Examples of `template` code have been given. It's really quite a simple function.

### `(templates` name arguments form `)`
This does not quite work as well as it should yet <sup><sub>**TODO**</sub></sup> It's meant to work on lists of arguments.

### `(cuda/shared` variable `)`
Creates a cuda `__shared__` variable.

## Binomial Operators

These include `+`, `-`, and the like. Each of these has a number of synonyms: These can take more than two arguments. For example,`(- a b c)` will come out to (after cleaning up the code) `(a-b)-c` or `a-b-c`. These are left or right reductive depending on whether they are in **C** or not. 

### `=`
This can be accessed through `=` `set` `let` `<-` and `:=`.
### `!=`
This can be accessed through `!=` `neq` `diff` and `different`.
### `==`
This can be accessed through `==` `eq` and `same`.
### `<`
This can be accessed through `<` and `lt`.
### `>`
This can be accessed through `>` and `gt`.
### `<=`
This can be accessed through `<=` `leq` and `le`.
### `>=`
This can be accessed through `>=` `geq` and `ge`.
### `&&`
This can be accessed through `&&` `and` `et` `und` and `y`.
### `&`
This can be accessed through `&` `bit-and` `band` `.and` `bit-et` `bet` `.et` `bit-und` `bund` `.und` `bit-y` `through` `.y` and ``.
### `&=`
This can be accessed through `&=` `&-eq` `bit-and-eq` `band-eq` `.and-eq` `bit-et-eq` `bet-eq` `.et-eq` `bit-und-eq` `bund-eq` `.und-eq` `bit-y-eq` `through-eq` `.y-eq` `&=` `bit-and=` `band=` `.and=` `bit-et=` `bet=` `.et=` `bit-und=` `bund=` `.und=` `bit-y=` `through=` `.y=` and ``.
### `||`
This can be accessed through `or` `uel` `oder` and `o`.
### `|`
This can be accessed through `bit-or` `.or` `bor` `bit-uel` `.uel` `buel` `bit-oder` `.oder` `boder` `bit-o` `.o` and `bo`.
### `|=`
This can be accessed through `bit-or-eq` `.or-eq` `bor-eq` `bit-uel-eq` `.uel-eq` `buel-eq` `bit-oder-eq` `.oder-eq` `boder-eq` `bit-o-eq` `.o-eq` `bo-eq` `bit-or=` `.or=` `bor=` `bit-uel=` `.uel=` `buel=` `bit-oder=` `.oder=` `boder=` `bit-o=` `.o=` and `bo=`.
### `+`
This can be accessed through `+` `plus` `add` and `sum`.
### `+=`
This can be accessed through `+=` `plus-eq` `add-eq` `sum-eq` `plus=` `add=` and `sum=`.
### `-`
This can be accessed through `-` `minus` `subtract` and `sub`.
### `-=`
This can be accessed through `-=` `minus-eq` `subtract-eq` `sub-eq` `minus=` `subtract=` and `sub=`.
### `*`
This can be accessed through `*` `times` `product` `mul` and `multiply`.
### `*=`
This can be accessed through `*=` `times-eq` `product-eq` `mul-eq` `multiply-eq` `times=` `product=` `mul=` and `multiply=`.
### `/`
This can be accessed through `/` `quotient` `ratio` `div` and `divide`.
### `/=`
This can be accessed through `/=` `quotient-eq` `ratio-eq` `div-eq` `divide-eq` `quotient=` `ratio=` `div=` and `divide=`.
### `%`
This can be accessed through `modulo` `mod` and `remainder`.
### `%=`
This can be accessed through `modulo-eq` `mod-eq` `remainder-eq` `modulo=` `mod=` and `remainder=`.
### `<<`
This can be accessed through `<<` `l-shift` `shift-left` and `shl`.
### `<<=`
This can be accessed through `<<=` `l-shift-eq` `shift-left-eq` `shl-eq` `l-shift=` `shift-left=` and `shl=`.
### `>>`
This can be accessed through `>>` `r-shift` `shift-right` and `shr`.
### `>>=`
This can be accessed through `>>=` `r-shift-eq` `shift-right-eq` `shr-eq` `>>=` `r-shift=` `shift-right=` and `shr=`.
### `->`
This can be accessed through `->` and `slot`.
### `.`
This can be accessed through `mem` and `.>`.

## Monomial Operators

These are operators that take in exactly one argument.

### `++` (pre)
This is the pre-increment (++x) operator. It can be accessed through `++` `inc` `+inc` `incr` `pre++` `+1` and `++n`.
### `++` (post)
This is the post-increment (x++) operator. It can be accessed through `+++` `pinc` `inc+` `pincr` `post++` `1+` and `n++`.
### `--` (pre)
This is the pre-decrement (--x) operator. It can be accessed through `--` `dec` `-dec` `decr` `pre--` `-1` and `--n`.
### `--` (post)
This is the post-decrement (x--) operator. It can be accessed through `---` `pdec` `dec-` `pdecr` `post--` `1-` and `n--`.
### `-`
This is the negation (-x) operator. It can be accessed through `neg`.
### `&`
This is the address-of (&x) operator. It can be accessed through `addr` `memloc` and `loc`.
### `!`
This is the not (!x) operator. It can be accessed through `!` `not` `un` `a` and `flip`.
### `~`
This is the bit-not (~x) operator. It can be accessed through `~` `bit-not` `bit-un` `bit-a` and `bit-flip`.

## What's With the Slashes?

You'll notice that `mpi/comm/size` compiles to `MPI_Comm_size` and that `cuda/dev->host` compiles to `cudaMemcpyDeviceToHost`. This is because external libraries are given support in this manner (with slashes).

## Synonyms

### General

| Term | Replacement |
| --- | --- |
| null | NULL |
| arg/c | argc |
| arg/count | argc |
| arg/v | argv |
| arg/values | argv |
| integer | int |
| integer+ | long |
| natural | unsigned int |
| natural+ | unsigned long |
| real | float |
| real+ | double |
| boolean | char |
| cstring | char* |

### CUDA

| Term | Replacement |
| --- | --- |
| cuda/malloc | cudaMalloc |
| cuda/memcpy | cudaMemcpy |
| cuda/free | cudaFree |
| cuda/host->dev | cudaMemcpyHostToDevice |
| cuda/dev->host | cudaMemcpyDeviceToHost |
| cuda/dev/count | cudaDeviceCount |
| cuda/dev/set | cudaSetDevice |
| cuda/dev/get | cudaGetDevice |
| cuda/dev/props | cudaDeviceProperties |
| cuda/sync | __syncthreads |
| block/idx | blockIdx |
| block/idx/x | blockIdx.x |
| block/idx/y | blockIdx.y |
| block/idx/z | blockIdx.z |
| thread/idx | threadIdx |
| thread/idx/x | threadIdx.x |
| thread/idx/y | threadIdx.y |
| thread/idx/z | threadIdx.z |
| block/dim | blockDim |
| block/dim/x | blockDim.x |
| block/dim/y | blockDim.y |
| block/dim/z | blockDim.z |
| grid/dim | gridDim |
| grid/dim/x | gridDim.x |
| grid/dim/y | gridDim.y |
| grid/dim/z | gridDim.z |
| dim/block | dimBlock |
| dim/grid | dimGrid |

### MPI 

| Term | Replacement |
| --- | --- |
| 
| mpi/success | MPI_SUCCESS |
| mpi/err/buffer | MPI_ERR_BUFFER |
| mpi/err/count | MPI_ERR_COUNT |
| mpi/err/type | MPI_ERR_TYPE |
| mpi/err/tag | MPI_ERR_TAG |
| mpi/err/comm | MPI_ERR_COMM |
| mpi/err/rank | MPI_ERR_RANK |
| mpi/err/request | MPI_ERR_REQUEST |
| mpi/err/root | MPI_ERR_ROOT |
| mpi/err/group | MPI_ERR_GROUP |
| mpi/err/op | MPI_ERR_OP |
| mpi/err/topology | MPI_ERR_TOPOLOGY |
| mpi/err/dims | MPI_ERR_DIMS |
| mpi/err/arg | MPI_ERR_ARG |
| mpi/err/unknown | MPI_ERR_UNKNOWN |
| mpi/err/truncate | MPI_ERR_TRUNCATE |
| mpi/err/other | MPI_ERR_OTHER |
| mpi/err/intern | MPI_ERR_INTERN |
| mpi/pending | MPI_PENDING |
| mpi/err/in/status | MPI_ERR_IN_STATUS |
| mpi/err/lastcode | MPI_ERR_LASTCODE |
| mpi/bottom | MPI_BOTTOM |
| mpi/proc/null | MPI_PROC_NULL |
| mpi/any/source | MPI_ANY_SOURCE |
| mpi/any/tag | MPI_ANY_TAG |
| mpi/undefined | MPI_UNDEFINED |
| mpi/bsend/overhead | MPI_BSEND_OVERHEAD |
| mpi/keyval/invalid | MPI_KEYVAL_INVALID |
| mpi/errors/are/fatal | MPI_ERRORS_ARE_FATAL |
| mpi/errors/return | MPI_ERRORS_RETURN |
| mpi/max/processor/name | MPI_MAX_PROCESSOR_NAME |
| mpi/max/error/string | MPI_MAX_ERROR_STRING |
| mpi/char | MPI_CHAR |
| mpi/short | MPI_SHORT |
| mpi/int | MPI_INT |
| mpi/long | MPI_LONG |
| mpi/unsigned/char | MPI_UNSIGNED_CHAR |
| mpi/unsigned/short | MPI_UNSIGNED_SHORT |
| mpi/unsigned | MPI_UNSIGNED |
| mpi/unsigned/long | MPI_UNSIGNED_LONG |
| mpi/float | MPI_FLOAT |
| mpi/double | MPI_DOUBLE |
| mpi/long/double | MPI_LONG_DOUBLE |
| mpi/byte | MPI_BYTE |
| mpi/packed | MPI_PACKED |
| mpi/float/int | MPI_FLOAT_INT |
| mpi/double/int | MPI_DOUBLE_INT |
| mpi/long/int | MPI_LONG_INT |
| mpi/2int | MPI_2INT |
| mpi/short/int | MPI_SHORT_INT |
| mpi/long/double/int | MPI_LONG_DOUBLE_INT |
| mpi/long/long/int | MPI_LONG_LONG_INT |
| mpi/ub | MPI_UB |
| mpi/lb | MPI_LB |
| mpi/comm/world | MPI_COMM_WORLD |
| mpi/comm/self | MPI_COMM_SELF |
| mpi/ident | MPI_IDENT |
| mpi/congruent | MPI_CONGRUENT |
| mpi/similar | MPI_SIMILAR |
| mpi/unequal | MPI_UNEQUAL |
| mpi/tag/ub | MPI_TAG_UB |
| mpi/io | MPI_IO |
| mpi/host | MPI_HOST |
| mpi/wtime/is/global | MPI_WTIME_IS_GLOBAL |
| mpi/max | MPI_MAX |
| mpi/min | MPI_MIN |
| mpi/sum | MPI_SUM |
| mpi/prod | MPI_PROD |
| mpi/maxloc | MPI_MAXLOC |
| mpi/minloc | MPI_MINLOC |
| mpi/band | MPI_BAND |
| mpi/bor | MPI_BOR |
| mpi/bxor | MPI_BXOR |
| mpi/land | MPI_LAND |
| mpi/lor | MPI_LOR |
| mpi/lxor | MPI_LXOR |
| mpi/group/null | MPI_GROUP_NULL |
| mpi/comm/null | MPI_COMM_NULL |
| mpi/datatype/null | MPI_DATATYPE_NULL |
| mpi/request/null | MPI_REQUEST_NULL |
| mpi/op/null | MPI_OP_NULL |
| mpi/errhandler/null | MPI_ERRHANDLER_NULL |
| mpi/group/empty | MPI_GROUP_EMPTY |
| mpi/graph | MPI_GRAPH |
| mpi/cart | MPI_CART |
| mpi/aint | MPI_Aint |
| mpi/status | MPI_Status |
| mpi/status/ignore | MPI_STATUS_IGNORE |
| mpi/statuses/ignore | MPI_STATUSES_IGNORE |
| mpi/group | MPI_Group |
| mpi/comm | MPI_Comm |
| mpi/datatype | MPI_Datatype |
| mpi/request | MPI_Request |
| mpi/op | MPI_Op |
| mpi/copy/function | MPI_Copy_function |
| mpi/delete/function | MPI_Delete_function |
| mpi/handler/function | MPI_Handler_function |
| mpi/user/function | MPI_User_function |
| mpi/init | MPI_Init |
| mpi/comm/size | MPI_Comm_size |
| mpi/comm/rank | MPI_Comm_rank |
| mpi/abort | MPI_Abort |
| mpi/get/processor/name | MPI_Get_processor_name |
| mpi/get/version | MPI_Get_version |
| mpi/initialized | MPI_Initialized |
| mpi/wtime | MPI_Wtime |
| mpi/wtick | MPI_Wtick |
| mpi/finalize | MPI_Finalize |

## Philosophy, Terminology, and Semiotics

The reason why __LISP__ is capitalized in __LISP__/**c** and **C** is not is because it looks more like __LISP__ than __C__. That's literally the whole reason.

The reason why I keep bolding __LISP__ and __C__ is for quick reference: the __LISP__-heavy portions and __C__-heavy portions of this document are intended to be useful to be able to be looked up.

__LISP__/__c__ is meant to be pronounced "lispsy".


## TODO

Add support for error checking.

Add transparent support for **C++**.

Add support for **OpenMP**.

Add support for handling `import` directives that span multiple directories.

Maybe get away from the `name-c` pragma.
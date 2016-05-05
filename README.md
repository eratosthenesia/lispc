# __LISP__/__c__

## __Intro__/duction
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

## **Why**?

Because __LISP__ is expressive and __C__ is fast and I wanted the best of both worlds is the short answer. The longer answer has something to do with macros. But instead of immediately boring you with that, I'll answer what you really want to know:

## **Why**/me?

First let's discuss if you *can* use it. Not to be elitist (I wish everyone would use this tool), but you **must** know both **c** and **LISP** fairly well to be able to use __LISP__/__c__.

Suppose, however, that you do already know both __LISP__ and __C__ pretty well. You might want to use __LISP__/__c__ because it features access to __LISP__ to write __C__ code obth implicitly and explicity. You might also want to use it if you like writing **CUDA** code, because it has built-in support for **CUDA** as well.

But really, to see why you might like to use __LISP__/__c__, check out a few examples, and feel free to skip around a little.

## **An**/Example

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

### __Arith__/metic

In the amove example, you'll notice that we use prefix arithmetic. This is a feature of **LISP** and not of **C**. The benefit of using prefix arithmetic is that it allows you to express sums of many terms somewhat more succinctly. That is to say, instead of
`2 + 3 + 4 + 5 * 6 + 7 + 8` you can just write `(+ 2 3 4 (* 5 6) 7 8)`.

### __Func__/tions

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

    (@foo 2 3 4)

### __Thing__/Names

Variable, type, function, etc. (identifier) names are converted using some simple rules. First of all, the `-`s are turned into `_`s. Secondly, everything is lower case. If you need to make something upper case entirely, you can prefix it with a `!` (so if you need the variable name `CONST_FOO` you can use `!const-foo`, `!cOnST-FoO`, `!const_FOO` or `"CONST_FOO"`). The last one may be used because strings are preserved. The others work because **LISP** is not case-sensitive, so when the variables are rendered, all case is the same. So if you were to use `cOnST-FoO` instead of `!cOnST-FoO`, you'd wind up with `const_foo` instead of `CONST_FOO`.

### __Continuing__/Forward

We need some sort of framework for showing each of the features of __LISP__/__c__. So before I go through every function and explain what it does, I'm going to explain a little of what goes on behind the scenes.

## __Eng__/ine

The main file for interacting with __LISP__/__c__ right now is just using __CLISP__ (for the time being; it will be ported to more versions soon) and typing in `(load "c.lisp")`.

You can test out the engine by either loading in a file using the `cwf` command and typing `(cwf "example.cl")`. What you'll see is either an error (because of syntax) or the resultant **C** code. If you don't have a file that you can experiment with yet, try typing the following:

	 (c '(typedef (struct foo (
			 bar
			 ((pt baz) float) )) qux))
It will result in the following (or similar):

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

## __Phil__/osophy, __Term__/inology, and __Sem__/iotics

The reason why __LISP__ is capitalized in __LISP__/**c** and **C** is not is because it looks more like __LISP__ than __C__. That's literally the whole reason.

The reason why I keep bolding __LISP__ and __C__ is for quick reference: the __LISP__-heavy portions and __C__-heavy portions of this document are intended to be useful to be able to be looked up.

__LISP__/__c__ is meant to be pronounced "lispsy".

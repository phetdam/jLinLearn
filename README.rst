.. README.rst for jLinSVM

jLinSVM
=======

A Java implementation of linear support vector machine models for regression and
classification of in-memory data sets. I needed to get myself reacquainted with
Java, which I last used two years ago, for one of my graduate classes, so I
figured this would be the perfect way to refresh myself while doing something
fun at the same time.

Although I may not be able to make significant progress during the school year,
I do intend to finish this project since it will be a nice exercise for me to
refresh my knowledge on some linear supervised learning methods.

Optimization methods
--------------------

A list of the optimization methods I intend to implement. This list won't be
exhaustive and these methods won't be state of the art by any means, but the
methods are still respectable.

Batch subgradient descent
   Will support :math:`l_2` regularization and :math:`l_1` + naive elastic net
   regularization using iterative soft thresholding [#]_. For those who are
   interested, there is a derivation of the soft thresholding operator from
   the promixal mapping in the first answer to
   `this question on the mathematics StackExchange`_.
Stochastic subgradient descent
   Will alse support :math:`l_1`, :math:`l_2`, and naive elastic net
   regularization using the same methods mentioned above for batch gradient
   descent.

.. [#] http://www.stat.cmu.edu/~ryantibs/convexopt/lectures/prox-grad.pdf

.. __: https://math.stackexchange.com/questions/471339/derivation-of-soft-
   thresholding-operator-proximal-operator-of-l-1-norm

These both will direcly solve the primal formulation of the problem by operating
on the loss functional directly. Not sure if I plan to implement any methods
to solve the dual problem.
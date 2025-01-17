# Building a basic SAT solver

> Written by Luka Bijma, published to [my blog](https://blazing-blast.github.io/) at 17th of January 2025.
> Writing started at the 16th.

## Foreword

My target audience consists of computer scientists, therefore I will be foregoing some of the mathematical rigor for the sake of keeping this readable to them.
In my notation, I will use both `!` as $\lnot$ for NOT, `&&` and $\land$ for AND, and `||` and $\lor$ for OR, depending on whether I am discussing theory or code.
I will assume these boolean operators as preliminary knowledge together with big $\mathcal{O}$ notation.

## What even is a SAT solver?

SAT stands for ‘satisfiability.’ A SAT solver’s job then is to find out whether a certain logic formula is satisfiable (SAT) or not (UNSAT). The formula will depend on several boolean variables, and it will be SAT if there exists an assignment of `true` and `false` to these variables such that the formula evaluates as `true`. First, let us discuss what this formula will even look like, and how to evaluate it.

### Boolean logic 101

Many programming languages implement so-called ‘boolean operators’. `&&` for AND, `||` for OR, and `!` for NOT. We will only consider these logical operators, and not others like $\implies$ and $\iff$. You may have heard that a computer can be build using just NAND-gates, as we have both AND and NOT, we also have NAND. Therefore, you can express all logic in these operators as well. In a proof that is left to the reader, you can find that all statements can be brought to an even more restricted form called Conjunctive Normal Form (CNF).

A formula in CNF can be described as a ‘conjunction of disjunctions,’ or more casually as an ‘AND of ORs.’ A disjunction is a bunch of other formulae linked together by ORs. In this case these other formulae must all be either $x$ or $\lnot x$ where $x$ is a boolean variable. This means that a formula in CNF is a bunch of these disjunctions linked together by ANDs. See here an example of a formula in CNF in math mathematical and programmer notation
$$(a \lor b) \land (a \lor \lnot b) \land (\lnot a \lor b),$$
$$(a \| b)\, \&\& \,(a \| !b)\, \&\& \,(!a \| b).$$

To evaluate such a formula, you simply fill in `true` or `false` for each of the variables, and apply the boolean operators.

This notation is useful, as it allows us to more efficiently solve the whether or not it is SAT. This is foreshadowing.

### The premise of a solver

A solver is a formula, and some info about it for parsing purposes. It should parse this formula, and then try combinations of variables until it finds one that works, then output SAT along with the variable assignment it found, if instead it has exhausted every possibility and still not found a solution, it should output UNSAT.

### The importance of SAT solvers

SAT solvers can be used to invert algorithms. As computers can be seen as a network or NAND gates, any algorithm with a fixed-length input that runs on a computer can be simulated using boolean logic. We can therefore create a SAT problem for such algorithm that is satisfied exactly when the algorithm gives a predetermined output given that the problem variables are used as algorithm arguments. This formula can then be solved to determine which input produces that fixed output. In other words, the SAT solver is like an algorithm inverter. If we can make a fast SAT solver, we can very quickly invert any such algorithm, given that we know its translation into CNF.

There are many more wonderful thing a SAT solver can do, but even just this is huge.

## The naive implementation

A formula will be expressed as follows. These definitions follow clearly from the Boolean Logic 101 section. Some SAT solvers put restrictions on the amount of terms in a disjunction, and can therefore use a statically-sized array. We will not make this assumption and need to deal with `Vector`s instead.

```Rust
struct CNF {
    disjunctions: Vec<Disjunction>,
}

struct Disjunction {
    terms: Vec<Term>,
}

struct Term {
    index: usize,
    negated: bool,
}
```

For now, we naively define the variable assignment that the solver is currently testing as

```Rust
struct Solver {
    formula: CNF,
    var_data: Vec<bool>, // vec![false, var_num]
    var_index
}
```

And then we implement an `evaluate` function as follows.

```Rust
impl CNF {
    fn evaluate(&self, solver: &Solver) -> bool {
        for disjunction in self.disjunctions {
            if (!disjunction.evaluate(solver))
            {
                return false;
            }
        }
        true
    }
}
```

You can guess how the implementation for the other classes looks yourself. Finally, we will implement the `Solver` class, which will be used to solve the formula.

```Rust
impl Solver {
    // returns
    // - Some(vars),    if SAT,
    // - None,          if UNSAT.
    fn solve(&mut self) -> Option<Variables> {
        if self.var_index >= self.data.len() {
            return match self.formula.evaluate() {
                true => Some(self.var_data),
                false => None,
            }
        }

        self.var_data[self.var_index] = true;
        self.var_index += 1;
        match solve() {
            Some(v) => return Some(v),
            None => (),
        }
        self.var_data[self.var_index - 1] = false;
        let out = solve();
        self.var_index -= 1;
        out;
    }
}
```

You can see that this algorithm is not very efficient. It recurses twice, and only solves at full depth.

## Undecidability as a tool for branch-elimination

This can be sped up using some tricks, but they do require a slight overhaul of our solvestates. We must embrace undecidability. When only some of our variables are assigned, some disjunctions may evaluate `true` or `false`, but other will be `unknown`. As soon as one of the disjunctions turns false, we already know that our branch has no hope of being SAT, as `false && x == false`. This would allow us to discard branches of the recursion tree much earlier and speed up the process by a lot. This is why CNF is so useful for SAT solving.

We can achieve this by modifying the `Solver` struct to the following

```Rust
// We could just use Option<bool>, but I believe this is clearer.
// In case you aren't familiar with Rust's enums, you can think of it as a C-style enum together with a union, also called a tagged union.
enum SolveState {
    Decided {
        value: bool,
    },
    Undecided
}

use SolveState::*;

struct Solver {
    formula: CNF,
    var_data: Vec<SolveState>, // vec![Undecided, var_num]
    var_index: usize,
}
```

We no longer need to use the index to keep track of which variable is set by which of the enum variants it is, instead, we simply use it to not have to recalculate which variable we're assigning too next.
With this new `SolveState`, we must update the `evaluate` functions.

```Rust
impl CNF {
    fn evaluate(&self, solver: &Solver) -> SolveState {
        let mut decided = true;
        for disjunction in self.disjunctions {
            match disjunction.evaluate(solver) {
                Decided(false) => return false,
                Decided(true) => (),
                Undecided => decided = false,
            }
        }
        match decided {
            false => Undecided,
            true => Decided(true),
        }
    }
}
```

The other implementations follow a similar pattern. With these changes (that impact performance for the worse), we can finally make changes that improve performance. We update the `solve` method with some branch elimination.

```Rust
impl Solver {
    fn solve(&mut self) -> Option<Variables> {
        // If we already know the result of this branch, we need not check it any further.
        match self.formula.evaluate(self.var_data) {
            Decided(false) => return None,
            Decided(true) => return Some(self.var_data),
            Undecided => (),
        }
        // This part is the same as last time.
        self.var_data[self.var_index] = Decided(true);
        self.var_index =+ 1;
        match solve() {
            Some(v) => return Some(v),
            None => (),
        }
        self.var_data[self.var_index - 1] = Decided(false);
        let out = solve();
        self.var_index -= 1;
        out;
    }
}
```

As we now return an `Option<Variables>` and `Variables` is a `Vec<SolveState>`, we might be left with some `Undecided` variables in out output. This means that these are free variables. This might be valuable output to some users. If you were to implement this yourself, it is up to you to output whether or not you fill the free variables with some truth value.

## Backtracing is more powerful than you think

Remember that we are using a recursive approach, you could already consider calling it backtracking, as we ‘backtrack’ the incrementation of `self.var_index`, but this is definitely not the full extend of the power of backtracking. We can consider the current level of recursion as the ‘real one.’ This means that we no longer modify the `&self`, but instead create a copy that we work on for future branches. This allows us to simplify the formula as we fill in more variables, which should speed up work. Consider the following `simplify` method. Note that is also returns a value just as `evaluate`, and modifies the input formula to the simpler variant.

```Rust
impl CNF {
    fn simplify(&mut self, solver: &mut Solver) {
        let mut decided = true;
        let mut next_disjunctions = Vec::new();
        for disjunction in self.disjunctions.iter_mut() {
            match disjunction.simplify(solver) { // Note that we call `simplify` here, not `evaluate`.
                Decided(false) => return false, // No simplification needed, as we will discard this branch anyways
                Decided(true) => (),
                Undecided => next_disjunctions.push(*disjunction),
            }
        }
        // This means that all disjunctions were true.
        if next_disjunctions.is_empty() {
            // No simplification is needed, as the branch is already solved
            return Decided(true);
        }
        // Filter out all the satisfied disjunctions
        // For this we should first derive `Eq` for Disjunction and Term, but just imagine that we did.
        next_disjunctions.dedup();
        self.disjunctions = next_disjunctions;
        Undecided
    }
}
```

As usual, this same logic can be extended to to `Disjunction::simplify`. On the other hand, `Term`s do not need to be simplified, as they will simply be eliminated from the disjunction as soon as they are resolved. We can now insert this new `simplify` method into our `solve` function (yes, it is a function now, because we're backtracking properly). This new implementation looks as follows:

```Rust
impl Solver {
    fn solve(&self) -> Option<Variables> {
        let mut next = self.clone();
        // If we already know the result of this branch, we need not check it any further.
        // We call this on `next` and not `self`, because we no longer wish to modify the input data
        match next.formula.simplify(self.var_data) {
            Decided(false) => return None,
            Decided(true) => return Some(self.var_data),
            Undecided => (),
        }

        next.var_index =+ 1;
        next.var_data[next.var_index] = Decided(true);
        match next.solve() {
            Some(v) => return Some(v),
            None => (),
        }
        next.var_data[next.var_index - 1] = Decided(false);
        next.solve()
        // We no longer need to decrement `var_index`, because we are using `next`, not `self`
    }
}
```

This will of course have higher memory usage, but the recursion depth never exceeds the amount of input variables, so that is negligible compared to the amount of computation we need. We do however also have more memory reads/writes, this can be a performance detriment, but we are also avoiding a lot of memory access by no longer having to evaluate certain disjunctions and terms. As the amount of leaves is exponential with the amount of variables, so is the cost of copying in the absolute worst case, but in most practical problems it will be quadratic to linear. This means that the amount of memory access we avoid is nigh guaranteed for the shallower levels of recursion. It might be beneficial to performance to only simplify up until a certain recursion depth, but that is for you to test.

## The order of variable resolution matters

In practical problems, there will always be a few disjunctions that only use a few terms. This means that if we can identify these, we could be able to target the variables in these terms. This would lead to eliminating these branches way earlier. You could instead target the most-used variables, but this will lead to more scattered memory writes and it works on reducing the impact of disjunction resolution, which are continuous in memory, and therefore likely have less of an impact on performance impact than eliminating a disjunction entirely (even if it is one disjunction vs many terms). This is speculation, as I have not found enough complicated SAT problems to properly performance-test my solver.

To do this, we sort the disjunctions by the amount of terms at the end of `CNF::simplify`. This means adding the following line

```Rust
next_disjunctions.sort_by(|x, y| x.terms.len().cmp(&y.terms.len()));
```

Then, we simply pick the first variable we find as the new index in `Solver::solve`. This also means that we no longer need to keep track of `Solver::var_index`. The new `solve` method is the following

```Rust
impl Solver {
    fn solve(&self) -> Option<Variables> {
        let mut next = self.clone();
        match next.formula.simplify(self.var_data) {
            Decided(false) => return None,
            Decided(true) => return Some(self.var_data),
            Undecided => (),
        }
        // These fields are not marked as public yet, but imagine that we fixed that in the mean time.
        let var_index = next.formula[0].terms[0].index;
        next.var_data[var_index] = Decided(true);
        match next.solve() {
            Some(v) => return Some(v),
            None => (),
        }
        next.var_data[var_index] = Decided(false);
        next.solve()
    }
}
```

This is the final improvement that I will show you, but I hope that you take away that there is a lot to be learned about SAT solvers, and that they are a really interesting tool to have, even for non-mathematicians. But most importantly, that they are demystified and have become a concrete piece of software to you, instead of just a weird, mathematical concept that has something to do with $P = NP$.

## Other things that I want to mention

### Input handling

You also need to be able to take input. (UN)SAT files are generally provided in the following format

```SAT
c `c` stands for comment
c `p` stands for parameters
c `cnf` specifies the notation. We only consider the CNF case.
p cnf VAR_NUM DISJUNCTION_NUM
DISJUNCTION
DISJUNCTION
DISJUNCTION
...
```

Where each disjunction is a space-separated list of numbers(variable names) and `-` stands for `NOT`. I will not consider myself with parsing this file, as it is not particularly engaging.

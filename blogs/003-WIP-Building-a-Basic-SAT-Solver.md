# Building a basic SAT solver

> A work-in-progress by Luka Bijma

## Foreword

Note that my target audience consists of computer scientists, therefore I will be foregoing some of the mathematical rigor for the sake of keeping this readable to them. I might add a ‘math nerding’ section if I have enough to discuss there.

In my notation, I will use both `!` as $\lnot$ for NOT, `&&` and $\land$ for AND, and `||` and $\lor$ for OR, depending on whether I am discussing theory or code.

I will assume these boolean operators as preliminary knowledge together with big $\mathcal{O}$ notation.

## What even is a SAT solver?

SAT stands for ‘satisfiability.’ A SAT solver's job then is to find out whether a certain logic formula is satisfiable (SAT) or not (UNSAT). The formula will depend on several boolean variables, and it will be SAT if there exists an assignment of TRUE and FALSE to these variables such that the formula evaluates as TRUE. First, let us discuss what this formula will even look like, and how to evaluate it.

### Boolean logic 101

Many programming languages implement so-called 'boolean operators'. `&&` for AND, `||` for OR, and `!` for NOT. We will only consider these logical operators, and not others like $\implies$ and $\iff$. From this, we can build any logical formula. You may have heard that a computer can be build using just NAND-gates, as we have both AND and NOT, we also have NAND. Therefore, you can express all logic in this form as well. In a proof that is left to the math section, you can find that all statements can be brought to an even more restricted form called Conjunctive Normal Form (CNF).

A formula in CNF can be described as a ‘conjunction of disjunctions,’ or more casually as an ‘AND of ORs.’ A disjunction is a bunch of other formulae linked together by ORs. In this case these other formulae must all be either $x$ or $\lnot x$ where $x$ is a boolean variable. This means that a formula in CNF is a bunch of these disjunctions linked together by ANDs.

To evaluate such a formula, you simply fill in TRUE or FALSE for each of the variables, and apply the boolean operators.

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

## Making it faster

This can be sped up using some tricks, but they do require a slight overhaul of our solvestates. We must embrace undecidability. When only some of our variables are assigned, some disjunctions may evaluate `true` or `false`, but other will be `unknown`. As soon as one of the disjunctions turns false, we already know that our branch has no hope of being SAT, as `false && x == false`. This would allow us to discard branches of the recursion tree much earlier and speed up the process by a lot. This is why CNF is so useful for SAT solving.

## Math Nerding

### Prove that DNF is complete

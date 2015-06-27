module Coins

open MicrosoftResearch.Infer.Fun.FSharp.Syntax

[<ReflectedDefinition>]
let coins () =
    let c1 = random (Bernoulli(0.5))
    let c2 = random (Bernoulli(0.5))
    let bothHeads = c1 && c2
    let bothTails = (not c1) && (not c2)
    observe (bothHeads = false && bothTails = false)
    c1, c2, bothHeads

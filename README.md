Last time: Created the ActionSpace class. 

Next time: It needs to be linked to the subject, and initialized when the subject is created (i.e. create space). Save updating the space for later.

After that... start working on how a subject can perform an action using that ActionSpace.


General sequence:

- Genetics is immutable, ActionSpace is mutable (changes over time depending on... things).
- Subject chooses an action within it's personal bounds (ActionSpace).. at first randomly.
- Performance varies below the maximum potential depending on parameters (maybe? keep it simple at first).. separate capability/potential from performance.
- Actions MUST have consequences. Consequences are derived from the ActionSpace but monitored through...????... need to figure this out. I don't like the idea of Health/Vitality/Mana, etc.
- Also... note..

Create - New structure
Destory - Damage structre
Consume - Receive value into actor
Produce - Give value from actor

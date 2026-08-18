# modules/

Nextflow DSL2 process definitions. Each file is one logical pipeline stage.

```
modules/local/     processes calling local Python scripts (no container registry dependency)
```

See `workflow/main.nf` for how processes are composed and `config/` for runtime parameters.

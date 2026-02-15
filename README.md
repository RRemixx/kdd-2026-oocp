# Title TODO

## Experiments
### Datasets (index)
1. markovar
2. flusight
3. iidlr
4. lane

### Naming of parameter files, config ids and experiment ids
1. Parameter files: <dataset_idx>*100 + [0-99]. The best one should be <dataset_idx>*100.
2. Config ids: <dataset_idx>*100 + [0-99]. The main results should be <dataset_idx>*100.
3. Experiment ids: Should be the same as [0-99] part. If multiple ones are the same, use index * 100 + [0-99].


### Main experiments
41: markovar,
44: lane,
45: cyclone,
46: flu_hosp,
47: weather,
48: elec

### Case study
#### Consistency (3000+)
On flu: 56
On weather: 57
On elec: 58 
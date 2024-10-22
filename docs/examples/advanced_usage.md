# Advanced Usage

Learn how to use MCBS for more complex scenarios:

## Purpose-Specific Models
```python
def purpose_specific_model(data):
    # Filter for specific trip purpose
    commuter_data = data[data['PURPOSE'] == 1]
    
    # Model specification
    ...
```

## Custom Metrics
```python
def custom_metric(results):
    return {
        'my_metric': calculate_my_metric(results)
    }

benchmark.run(models, additional_metrics=[custom_metric])
```
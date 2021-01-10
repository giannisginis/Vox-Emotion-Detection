def memory_usage_psutil(unit="MB"):
    coeff = 0.0
    # return the memory usage in MB
    import resource
    if unit == "MB":
        coeff = 1000.0
    elif unit == "GB":
        coeff = 1000.0 * 1000.0
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / coeff
    # print 'Memory usage: %s (MB)' % (mem)
    return mem

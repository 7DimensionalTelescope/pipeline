def autoproperties(cls):
    for attr in list(vars(cls)):
        if attr.startswith('_') and not attr.startswith('__'):
            prop_name = attr[1:]
            if not hasattr(cls, prop_name):
                # Define property and setter using closure
                def make_property(attr_name):
                    return property(
                        lambda self: getattr(self, attr_name),
                        lambda self, value: setattr(self, attr_name, value)
                    )
                setattr(cls, prop_name, make_property(attr))
    return cls


class classmethodproperty:
    """
    A custom decorator that combines class method and property behaviors.

    Allows creating class-level properties that can be accessed 
    without instantiating the class, while maintaining the 
    flexibility of class methods.

    Typical use cases:
    - Generating computed class-level attributes
    - Providing dynamic class-level information
    - Implementing lazy-loaded class properties

    Attributes:
        func (classmethod): The underlying class method

    Example:
        class Example:
            @classmethodproperty
            def dynamic_property(cls):
                return compute_something_for_class()
    """

    def __init__(self, func):
        """
        Initialize the classmethodproperty decorator.

        Args:
            func (callable): The function to be converted to a class method property
        """
        self.func = classmethod(func)
    
    def __get__(self, instance, owner):
        """
        Retrieve the value of the class method property.

        Args:
            instance: The instance calling the property (ignored)
            owner: The class on which the property is defined

        Returns:
            The result of calling the class method
        """
        return self.func.__get__(instance, owner)()
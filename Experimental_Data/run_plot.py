import os
import sys

root_path = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\'))
data_names = ['MNIST','MEDICAL', 'QUANTUM']

class run_plot:
    def __init__(self):
        pass
    
    def inputStr(self, title, default):
        result = input(title)
        if result == '':
            result = default
        return result
    
    def getChoice(self, keys, accept_empty=False, accept_range=False):
        """Get a keypress from the user from the specified keys."""
        r = ''
        possible_keys = keys.copy()
        if accept_range:
            for ki in keys:
                for kj in keys:
                    possible_keys.update({f'{ki}-{kj}': None})

        while not r or r not in possible_keys:
            r = input().upper()
            if accept_empty and r == '':
                break
        return r

    def displayOptions(self, options):
        keys = {}
        for i, opt in enumerate(options):
            key = '%d' % (i + 1)
            keys[key] = opt
            print(f'  [{key}] : {opt}')
        return keys

    def selectFromList(self,
                        options,
                        title,
                        all=True,
                        quit=True,
                        input=False,
                        accept_empty=False,
                        default=None):
        """Get a user-selected option
        
        Returns an element from options.
        """
        print()
        print()
        print(f'Select {title}')
        print()
        keys = self.displayOptions(options)
        if all:
            keys['A'] = 'All'
            print('  [A] : All')
        if quit:
            keys['Q'] = None
            print('  [Q] : Quit')
        if input:
            keys['I'] = 'Input'
            print('  [I] : Input')
        keys[''] = default
        k = self.getChoice(keys, accept_empty=accept_empty)
        return keys[k]

    def plot_figure(self, data_name):
        code_path = root_path+'\\figure_'+data_name+'\\code'
        if code_path in sys.path:
            sys.path.remove(code_path)
        sys.path.insert(0, code_path)
        if data_name == 'MEDICAL':
            import plot_medical
            plot_medical.run()
        elif data_name == 'QUANTUM':
            import plot_quantum
            plot_quantum.run()
        else:
            import plot_mnist
            plot_mnist.run()
        sys.path.remove(code_path)

    def run(self):
        while True:
            self.selected_data = self.selectFromList(options=data_names, title='select a data type to plot: ')
            if self.selected_data is None:
                break
            elif self.selected_data == 'All':
                for data_name in data_names:
                    self.plot_figure(data_name)
            else:
                self.plot_figure(self.selected_data)

    
if __name__ == '__main__':
    run_plot().run()
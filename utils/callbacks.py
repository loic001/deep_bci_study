from skorch.callbacks import Callback
import pickle
from skorch.utils import noop
import warnings
class MyCheckpoint(Callback):
    """Save the model during training if the given metric improved.
    This callback works by default in conjunction with the validation
    scoring callback since it creates a ``valid_loss_best`` value
    in the history which the callback uses to determine if this
    epoch is save-worthy.
    You can also specify your own metric to monitor or supply a
    callback that dynamically evaluates whether the model should
    be saved in this epoch.
    Some or all of the following can be saved:
      - model parameters (see ``f_params`` parameter);
      - training history (see ``f_history`` parameter);
      - entire model object (see ``f_pickle`` parameter).
    You can implement your own save protocol by subclassing
    ``Checkpoint`` and overriding :func:`~Checkpoint.save_model`.
    This callback writes a bool flag to the history column
    ``event_cp`` indicating whether a checkpoint was created or not.
    Example:
    >>> net = MyNet(callbacks=[Checkpoint()])
    >>> net.fit(X, y)
    Example using a custom monitor where models are saved only in
    epochs where the validation *and* the train losses are best:
    >>> monitor = lambda net: all(net.history[-1, (
    ...     'train_loss_best', 'valid_loss_best')])
    >>> net = MyNet(callbacks=[Checkpoint(monitor=monitor)])
    >>> net.fit(X, y)
    Parameters
    ----------
    target : deprecated
    monitor : str, function, None
      Value of the history to monitor or callback that determines
      whether this epoch should lead to a checkpoint. The callback
      takes the network instance as parameter.
      In case ``monitor`` is set to ``None``, the callback will save
      the network at every epoch.
      **Note:** If you supply a lambda expression as monitor, you cannot
      pickle the wrapper anymore as lambdas cannot be pickled. You can
      mitigate this problem by using importable functions instead.
    f_params : file-like object, str, None (default='params.pt')
      File path to the file or file-like object where the model
      parameters should be saved. Pass ``None`` to disable saving
      model parameters.
      If the value is a string you can also use format specifiers
      to, for example, indicate the current epoch. Accessible format
      values are ``net``, ``last_epoch`` and ``last_batch``.
      Example to include last epoch number in file name:
      >>> cb = Checkpoint(f_params="params_{last_epoch[epoch]}.pt")
    f_history : file-like object, str, None (default=None)
      File path to the file or file-like object where the model
      training history should be saved. Pass ``None`` to disable
      saving history.
      Supports the same format specifiers as ``f_params``.
    f_pickle : file-like object, str, None (default=None)
      File path to the file or file-like object where the entire
      model object should be pickled. Pass ``None`` to disable
      pickling.
      Supports the same format specifiers as ``f_params``.
    sink : callable (default=noop)
      The target that the information about created checkpoints is
      sent to. This can be a logger or ``print`` function (to send to
      stdout). By default the output is discarded.
    """
    def __init__(
            self,
            target=None,
            monitor='valid_loss_best',
            f_params='params.pt',
            f_history=None,
            f_pickle=None,
            sink=noop,
            callback_func=None
    ):
        if target is not None:
            warnings.warn(
                "target argument was renamed to f_params and will be removed "
                "in the next release. To make your code future-proof it is "
                "recommended to explicitly specify keyword arguments' names "
                "instead of relying on positional order.",
                DeprecationWarning)
            f_params = target
        self.monitor = monitor
        self.f_params = f_params
        self.f_history = f_history
        self.f_pickle = f_pickle
        self.sink = sink
        self.callback_func = callback_func

    def on_epoch_end(self, net, **kwargs):
        if self.monitor is None:
            do_checkpoint = True
        elif callable(self.monitor):
            do_checkpoint = self.monitor(net)
        else:
            try:
                do_checkpoint = net.history[-1, self.monitor]
            except KeyError as e:
                raise SkorchException(
                    "Monitor value '{}' cannot be found in history. "
                    "Make sure you have validation data if you use "
                    "validation scores for checkpointing.".format(e.args[0]))

        if do_checkpoint:
            self.save_model(net)
            self._sink("A checkpoint was triggered in epoch {}.".format(
                len(net.history) + 1
            ), net.verbose)

        net.history.record('event_cp', bool(do_checkpoint))

    def save_model(self, net):
        """Save the model.
        This function saves some or all of the following:
          - model parameters;
          - training history;
          - entire model object.
        """
        if self.f_params:
            net.save_params(self._format_target(net, self.f_params))
            if self.callback_func and callable(self.callback_func):
                self.callback_func(self._format_target(net, self.f_params))
        if self.f_history:
            net.save_history(self._format_target(net, self.f_history))
        if self.f_pickle:
            f_pickle = self._format_target(net, self.f_pickle)
            with open_file_like(f_pickle, 'wb') as f:
                pickle.dump(net, f)

    def _format_target(self, net, f):
        """Apply formatting to the target filename template."""
        if isinstance(f, str):
            return f.format(
                net=net,
                last_epoch=net.history[-1],
                last_batch=net.history[-1, 'batches', -1],
            )
        return f

    def _sink(self, text, verbose):
        #  We do not want to be affected by verbosity if sink is not print
        if (self.sink is not print) or verbose:
            self.sink(text)

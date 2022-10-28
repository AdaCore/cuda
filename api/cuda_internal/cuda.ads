--  This is a temporary workaround to ensure that cuda.internal is
--  always linked in a CUDA application. Otherwise, the object is
--  only referred to in the binder and doesn't get added at link-time.
limited with CUDA.Internal;

package CUDA is

end CUDA;
(eval-when (:execute :compile-toplevel :load-toplevel)
  (ql:quickload '(cl-glfw cl-opengl cl-glu)))

(declaim (optimize (debug 3) (speed 1) (safety 3)))

(defpackage :disp
  (:use :cl :gl))

(in-package :disp)

(with-open-file (s "~/1122/opencv/build/eyepos")
  (loop for line = (read-line s nil nil)
       while line do
       (format t "~a~%" (read-from-string line))))


(let ((rot 0))
 (defun draw ()
   (load-identity)
   (glu:look-at 0 20 14 ;; cam
		0 0 0   ;; target
		0 0 1)
   (clear :color-buffer-bit)
   (clear-color 0 0 0 0)
   
   
   (begin :lines)
   (color 1 0 0) (vertex 0 0 0) (vertex  1  0 0)
   (color 0 1 0) (vertex 0 0 0) (vertex  0  1 0)
   (color 0 0 1) (vertex 0 0 0) (vertex  0  0 1)
   (end)
   
   
   ))

#+nil
(glfw:do-window (:title "A Simple cl-opengl Example")
    ((glfw:swap-interval 1)
     (matrix-mode :projection)
     (load-identity)
     (unwind-protect (glu:perspective 45 4/3 0.1 50)
       (matrix-mode :modelview)))
  (when (eql (glfw:get-key glfw:+key-esc+) glfw:+press+)
    (return-from glfw::do-open-window))
  (draw))

(eval-when (:execute :compile-toplevel :load-toplevel)
  (ql:quickload '(cl-glfw cl-opengl cl-glu)))

(declaim (optimize (debug 3) (speed 1) (safety 3)))

(defpackage :disp
  (:use :cl :gl))

(in-package :disp)

#+nil
(with-open-file (s "~/1122/opencv/build/eyepos")
  (loop for line = (read-line s nil nil)
       while line do
       (format t "~a~%" (read-from-string line))))

#+nil
(defparameter *eye-input* (open "~/1122/opencv/build/eyepos"))
#+nil
(read *eye-input*)
#+nil
(close *eye-input*)


(let ((rot 0))
 (defun draw ()
   ;;(format t "~a~%" (read *eye-input*))
   (load-identity)
   (let ((line (read *eye-input*)))
     
    (destructuring-bind (ex ey l) line
      (let* ((y (+ 20 (* .3 (- 100 l))))
	     (z (* -20 ey))
	     (x (* 10 ex)))
       (format t "~a~%" (list x y z)) 
       (glu:look-at x y z ;; cam
		    0 0 0			   ;; target
		    0 0 1))))
   (clear :color-buffer-bit)
   (clear-color 0 0 0 0)
   
   (scale 10 10 10)
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
